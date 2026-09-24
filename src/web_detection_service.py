"""Backend detection service for the Airacare web application."""

from __future__ import annotations

import base64
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from src.config import TARGET_CLASS_SET
from src.detector import (
    CONFIDENCE_THRESHOLD,
    CUSTOM_MODEL_PATH,
    INFERENCE_IMAGE_SIZE,
    IOU_THRESHOLD,
    load_detection_model,
    normalize_model_names,
    resolve_device,
)
from src.distance import DistanceHistory, estimate_distance, focal_length_for_runtime_resolution, load_calibration
from src.path_analysis import DEFAULT_PATH_CONFIG_PATH, PathAnalyzer
from src.relative_motion import CLOSING_THRESHOLD_MPS, RelativeMotionAnalyzer
from src.risk import RiskResult, RiskSmoother, evaluate_risk, scene_risk
from src.tracker import TrackHistory
from src.warning import NONE, RISK_TO_WARNING, WarningCandidate, WarningManager

logger = logging.getLogger("airacare-web-backend")


@dataclass
class WebDetectionConfig:
    model_mode: str = "custom"
    confidence: float = CONFIDENCE_THRESHOLD
    imgsz: int = INFERENCE_IMAGE_SIZE
    iou: float = IOU_THRESHOLD
    device: str = "auto"
    calibration_path: str = "config/webcam_calibration.json"
    path_config_path: str = str(DEFAULT_PATH_CONFIG_PATH)
    vehicle_speed_kmh: float = 30.0


@dataclass
class _TrackMemory:
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    last_seen: float = field(default_factory=time.perf_counter)


class WebDetectionService:
    """Stateful detector used by the FastAPI backend."""

    def __init__(self, config: WebDetectionConfig | None = None) -> None:
        self.config = config or WebDetectionConfig()
        self.model = None
        self.using_pretrained_fallback = False
        self.model_path = None
        self.model_names: dict[int, str] = {}
        self.device = resolve_device(self.config.device)
        self.lock = threading.Lock()
        self.path_analyzer = PathAnalyzer.from_file(self.config.path_config_path)
        self.distance_history = DistanceHistory()
        self.motion_analyzer = RelativeMotionAnalyzer()
        self.risk_smoother = RiskSmoother()
        self.warning_manager = WarningManager(warnings_enabled=True, audio_enabled=False)
        self.track_history = TrackHistory()
        self.web_tracks: list[_TrackMemory] = []
        self.next_track_id = 1
        self.calibration = load_calibration(self.config.calibration_path)
        self.runtime_focal_length: float | None = None
        self.runtime_focal_warning: str | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        if len(inspect.signature(load_detection_model).parameters) == 0:
            model, using_pretrained, model_path = load_detection_model()
        else:
            model, using_pretrained, model_path = load_detection_model(self.config.model_mode)
        if model is None:
            raise RuntimeError(f"Could not load Airacare model from {CUSTOM_MODEL_PATH}")
        self.model = model
        self.using_pretrained_fallback = using_pretrained
        self.model_path = model_path
        self.model_names = normalize_model_names(getattr(model, "names", {}))
        logger.info(
            "model loaded path=%s fallback=%s names=%s",
            self.model_path,
            self.using_pretrained_fallback,
            self.model_names,
        )

    def detect_data_url(self, image_data_url: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        frame = _decode_data_url(image_data_url)
        return self.detect_frame(frame, metadata or {})

    def detect_frame(self, frame: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        self.load()
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        with self.lock:
            frame_height, frame_width = frame.shape[:2]
            logger.info("image decoded width=%s height=%s", frame_width, frame_height)
            if self.calibration is not None and self.runtime_focal_length is None:
                self.runtime_focal_length, self.runtime_focal_warning = focal_length_for_runtime_resolution(
                    self.calibration,
                    (frame_width, frame_height),
                )

            results = self.model.predict(
                source=frame,
                conf=self.config.confidence,
                imgsz=self.config.imgsz,
                iou=self.config.iou,
                device=self.device,
                verbose=False,
            )
            now = time.perf_counter()
            result = results[0] if results else None
            boxes = getattr(result, "boxes", []) if result is not None else []
            detections: list[dict[str, Any]] = []
            risks: list[RiskResult] = []
            warning_candidates: list[WarningCandidate] = []

            self._cleanup_web_tracks(now)

            for detected_box in _class_agnostic_nms(list(boxes)):
                class_id = int(detected_box.cls[0])
                class_name = self.model_names.get(class_id, f"class_{class_id}").lower()
                if class_name not in TARGET_CLASS_SET:
                    continue

                confidence = float(detected_box.conf[0])
                x1, y1, x2, y2 = (float(value) for value in detected_box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)
                track_id = self._assign_track_id(class_name, bbox, now)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                self.track_history.update(track_id, class_id, class_name, confidence, bbox, center)

                path_zone = self.path_analyzer.classify(center, frame_width, frame_height)
                bbox_height = max(0.0, y2 - y1)
                raw_distance = (
                    estimate_distance(class_name, bbox_height, self.runtime_focal_length)
                    if self.runtime_focal_length is not None
                    else None
                )
                distance_m = self.distance_history.update(track_id, raw_distance)
                motion = self.motion_analyzer.update(track_id, distance_m, now)
                motion_state = motion.state
                if motion.closing_speed_mps is not None:
                    if motion.closing_speed_mps > CLOSING_THRESHOLD_MPS:
                        motion_state = "CLOSING"
                    elif motion.closing_speed_mps < -CLOSING_THRESHOLD_MPS:
                        motion_state = "OPENING"
                    else:
                        motion_state = "STABLE"

                raw_risk = evaluate_risk(
                    zone=path_zone,
                    distance_m=distance_m,
                    closing_speed_mps=motion.closing_speed_mps,
                    ttc_seconds=motion.ttc_seconds,
                    movement_state=motion_state,
                    detection_confidence=confidence,
                    vehicle_speed_kmh=float(metadata.get("vehicleSpeedKmh") or self.config.vehicle_speed_kmh),
                )
                smoothed_level = self.risk_smoother.update(track_id, raw_risk.level, now)
                risk_result = RiskResult(smoothed_level, raw_risk.score, raw_risk.reason, raw_level=raw_risk.level)
                risks.append(risk_result)
                warning_candidates.append(
                    WarningCandidate(
                        track_id=track_id,
                        class_name=class_name,
                        risk_level=smoothed_level,
                        zone=path_zone,
                        distance_m=distance_m,
                        ttc_seconds=motion.ttc_seconds,
                        reason=raw_risk.reason,
                    )
                )

                warning_level = RISK_TO_WARNING.get(smoothed_level, NONE)
                detections.append(
                    {
                        "trackId": track_id,
                        "label": class_name,
                        "classId": class_id,
                        "confidence": confidence,
                        "distanceMeters": distance_m,
                        "riskLevel": smoothed_level,
                        "rawRiskLevel": raw_risk.level,
                        "riskScore": raw_risk.score,
                        "riskReason": raw_risk.reason,
                        "warningLevel": warning_level,
                        "warningTriggered": warning_level != NONE,
                        "movementDirection": motion_state,
                        "closingSpeedMps": motion.closing_speed_mps,
                        "ttcSeconds": motion.ttc_seconds,
                        "pathZone": path_zone,
                        "boundingBox": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                    }
                )

            self.distance_history.cleanup_stale()
            self.motion_analyzer.cleanup_stale()
            self.risk_smoother.cleanup_stale()
            self.track_history.cleanup_stale()
            warning_state = self.warning_manager.update(warning_candidates, now)
            scene_level = scene_risk(risks)

            timestamp_ms = int(time.time() * 1000)
            for detection in detections:
                if detection["trackId"] == warning_state.track_id:
                    detection["warningMessage"] = warning_state.message
                    detection["audioTriggered"] = warning_state.audio_triggered
                else:
                    detection["warningMessage"] = ""
                    detection["audioTriggered"] = False
                detection.update(
                    {
                        "vehicleSpeedKmh": metadata.get("vehicleSpeedKmh", self.config.vehicle_speed_kmh),
                        "latitude": metadata.get("latitude"),
                        "longitude": metadata.get("longitude"),
                        "bearingDegrees": metadata.get("bearingDegrees"),
                        "cameraSource": "web_camera_backend",
                        "deviceType": metadata.get("deviceType", "Web"),
                        "detectedAtEpochMillis": timestamp_ms,
                        "timestamp": timestamp_ms,
                    }
                )

            return {
                "ok": True,
                "modelPath": str(self.model_path) if self.model_path else None,
                "usingPretrainedFallback": self.using_pretrained_fallback,
                "frameWidth": frame_width,
                "frameHeight": frame_height,
                "sceneRisk": scene_level,
                "warning": {
                    "level": warning_state.level,
                    "message": warning_state.message,
                    "trackId": warning_state.track_id,
                    "className": warning_state.class_name,
                    "riskLevel": warning_state.risk_level,
                    "distanceMeters": warning_state.distance_m,
                    "ttcSeconds": warning_state.ttc_seconds,
                },
                "detections": detections,
                "timestamp": timestamp_ms,
                "calibrationWarning": self.runtime_focal_warning,
            }

    def _assign_track_id(self, class_name: str, bbox: tuple[float, float, float, float], now: float) -> int:
        best_track = None
        best_overlap = 0.0
        for track in self.web_tracks:
            if track.class_name != class_name:
                continue
            overlap = _iou(track.bbox, bbox)
            if overlap > best_overlap:
                best_track = track
                best_overlap = overlap
        if best_track is not None and best_overlap >= 0.25:
            best_track.bbox = bbox
            best_track.last_seen = now
            return best_track.track_id

        track_id = self.next_track_id
        self.next_track_id += 1
        self.web_tracks.append(_TrackMemory(track_id=track_id, class_name=class_name, bbox=bbox, last_seen=now))
        return track_id

    def _cleanup_web_tracks(self, now: float) -> None:
        self.web_tracks = [track for track in self.web_tracks if now - track.last_seen <= 2.0]


def _decode_data_url(image_data_url: str) -> Any:
    if "," in image_data_url:
        image_data_url = image_data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data_url)
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Could not decode image frame")
    return frame


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _box_tuple(detected_box: Any) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in detected_box.xyxy[0].tolist())


def _class_agnostic_nms(boxes: list[Any], iou_threshold: float = 0.60) -> list[Any]:
    """Keep the strongest label when multiple classes overlap the same object."""
    sorted_boxes = sorted(boxes, key=lambda item: float(item.conf[0]), reverse=True)
    kept: list[Any] = []
    kept_boxes: list[tuple[float, float, float, float]] = []
    for candidate in sorted_boxes:
        candidate_box = _box_tuple(candidate)
        if any(_iou(candidate_box, existing) >= iou_threshold for existing in kept_boxes):
            continue
        kept.append(candidate)
        kept_boxes.append(candidate_box)
    return kept
