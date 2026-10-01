"""Backend detection service for the Airacare web application."""

from __future__ import annotations

import base64
import inspect
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import cv2
import numpy as np

from src.config import TARGET_CLASS_SET
from src.android_tflite_detector import ANDROID_TFLITE_MODEL_PATH, AndroidTfliteDetector
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

CLASS_CONFIDENCE_THRESHOLDS = {
    "person": float(os.getenv("AIRACARE_PERSON_CONFIDENCE", "0.55")),
    "dog": float(os.getenv("AIRACARE_DOG_CONFIDENCE", "0.65")),
    "cat": float(os.getenv("AIRACARE_CAT_CONFIDENCE", "0.80")),
    "horse": float(os.getenv("AIRACARE_HORSE_CONFIDENCE", "0.65")),
    "cow": float(os.getenv("AIRACARE_COW_CONFIDENCE", "0.65")),
    "deer": float(os.getenv("AIRACARE_DEER_CONFIDENCE", "0.65")),
    "goat": float(os.getenv("AIRACARE_GOAT_CONFIDENCE", "0.70")),
    "elephant": float(os.getenv("AIRACARE_ELEPHANT_CONFIDENCE", "0.70")),
}
DEFAULT_CLASS_CONFIDENCE = float(os.getenv("AIRACARE_DEFAULT_CONFIDENCE", "0.65"))
MIN_CONSECUTIVE_FRAMES = int(os.getenv("AIRACARE_MIN_CONFIRMATION_FRAMES", "3"))
MIN_BOX_AREA_RATIO = float(os.getenv("AIRACARE_MIN_BOX_AREA_RATIO", "0.0025"))
MAX_BOX_AREA_RATIO = float(os.getenv("AIRACARE_MAX_BOX_AREA_RATIO", "0.85"))
MAX_OUTSIDE_RATIO = float(os.getenv("AIRACARE_MAX_BOX_OUTSIDE_RATIO", "0.10"))
MIN_BOX_DIMENSION_PX = float(os.getenv("AIRACARE_MIN_BOX_DIMENSION_PX", "8"))
TRACK_STALE_SECONDS = float(os.getenv("AIRACARE_WEB_TRACK_STALE_SECONDS", "8.0"))
QUALITY_MIN_BRIGHTNESS = float(os.getenv("AIRACARE_QUALITY_MIN_BRIGHTNESS", "18"))
QUALITY_MAX_BRIGHTNESS = float(os.getenv("AIRACARE_QUALITY_MAX_BRIGHTNESS", "238"))
QUALITY_MIN_CONTRAST = float(os.getenv("AIRACARE_QUALITY_MIN_CONTRAST", "10"))
QUALITY_MIN_BLUR_VARIANCE = float(os.getenv("AIRACARE_QUALITY_MIN_BLUR_VARIANCE", "18"))
QUALITY_MIN_EDGE_RATIO = float(os.getenv("AIRACARE_QUALITY_MIN_EDGE_RATIO", "0.002"))


def resolve_web_model_backend() -> str:
    requested = os.getenv("AIRACARE_MODEL_BACKEND", "android_tflite")
    if os.getenv("AIRACARE_FORCE_YOLO_PT") == "1":
        return "yolo_pt"
    if requested == "yolo_pt":
        logger.info("AIRACARE_MODEL_BACKEND=yolo_pt ignored for web stability; using android_tflite")
        return "android_tflite"
    return requested


@dataclass
class WebDetectionConfig:
    model_mode: str = "custom"
    confidence: float = CONFIDENCE_THRESHOLD
    imgsz: int = int(os.getenv("AIRACARE_INFERENCE_IMAGE_SIZE", "320"))
    iou: float = IOU_THRESHOLD
    device: str = "auto"
    calibration_path: str = "config/phone_calibration.json"
    path_config_path: str = str(DEFAULT_PATH_CONFIG_PATH)
    vehicle_speed_kmh: float = 30.0


@dataclass
class _TrackMemory:
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    last_seen: float = field(default_factory=time.perf_counter)
    first_seen: float = field(default_factory=time.perf_counter)
    consecutive_frames: int = 0
    max_confidence: float = 0.0
    confidence_sum: float = 0.0
    observation_count: int = 0


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
        self.runtime_focal_resolution: tuple[int, int] | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch

            torch.set_num_threads(int(os.getenv("AIRACARE_TORCH_THREADS", "1")))
            torch.set_num_interop_threads(int(os.getenv("AIRACARE_TORCH_INTEROP_THREADS", "1")))
        except Exception as error:
            logger.warning("could not tune torch threading: %s", error)

        backend = resolve_web_model_backend()
        model = None
        using_pretrained = False
        model_path = None
        if backend != "android_tflite":
            if len(inspect.signature(load_detection_model).parameters) == 0:
                model, using_pretrained, model_path = load_detection_model()
            else:
                model, using_pretrained, model_path = load_detection_model(self.config.model_mode)
        elif ANDROID_TFLITE_MODEL_PATH.exists():
            model = AndroidTfliteDetector(num_threads=int(os.getenv("AIRACARE_TFLITE_THREADS", "4")))
            using_pretrained = False
            model_path = ANDROID_TFLITE_MODEL_PATH
        if model is None:
            raise RuntimeError(f"Could not load Airacare model from {CUSTOM_MODEL_PATH}")
        self.model = model
        self.using_pretrained_fallback = using_pretrained
        self.model_path = model_path
        self.model_names = normalize_model_names(getattr(model, "names", {}))
        logger.info(
            "model loaded backend=%s path=%s fallback=%s names=%s inference_imgsz=%s",
            backend,
            self.model_path,
            self.using_pretrained_fallback,
            self.model_names,
            self.config.imgsz,
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
            frame_quality = _analyze_frame_quality(frame)
            if not frame_quality["ok"]:
                logger.warning(
                    "frame rejected quality=%s brightness=%.2f contrast=%.2f blur=%.2f edgeRatio=%.5f",
                    frame_quality["reason"],
                    frame_quality["brightness"],
                    frame_quality["contrast"],
                    frame_quality["blurVariance"],
                    frame_quality["edgeRatio"],
                )
                self._reset_transient_state()
                return self._empty_result(frame_width, frame_height, frame_quality)

            runtime_resolution = (frame_width, frame_height)
            if self.calibration is not None and self.runtime_focal_resolution != runtime_resolution:
                self.runtime_focal_length, self.runtime_focal_warning = focal_length_for_runtime_resolution(
                    self.calibration,
                    runtime_resolution,
                )
                self.runtime_focal_resolution = runtime_resolution
                logger.info(
                    "distance calibration runtime=%s focal=%.2f warning=%s",
                    runtime_resolution,
                    self.runtime_focal_length,
                    self.runtime_focal_warning,
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
            logger.info("raw model boxes count=%s modelPath=%s", len(boxes), self.model_path)
            detections: list[dict[str, Any]] = []
            risks: list[RiskResult] = []
            warning_candidates: list[WarningCandidate] = []

            self._cleanup_web_tracks(now)

            raw_boxes = list(boxes)
            for raw_box in raw_boxes:
                raw_class_id = int(raw_box.cls[0])
                raw_label = self.model_names.get(raw_class_id, f"class_{raw_class_id}").lower()
                raw_confidence = float(raw_box.conf[0])
                raw_bbox = tuple(float(value) for value in raw_box.xyxy[0].tolist())
                logger.info(
                    "raw detection modelPath=%s rawClassId=%s mappedLabel=%s confidence=%.4f bbox=%s",
                    self.model_path,
                    raw_class_id,
                    raw_label,
                    raw_confidence,
                    tuple(round(value, 1) for value in raw_bbox),
                )

            for detected_box in _class_agnostic_nms(raw_boxes):
                class_id = int(detected_box.cls[0])
                class_name = self.model_names.get(class_id, f"class_{class_id}").lower()
                if class_name not in TARGET_CLASS_SET:
                    logger.info(
                        "detection rejected class_not_target rawClassId=%s mappedLabel=%s modelPath=%s",
                        class_id,
                        class_name,
                        self.model_path,
                    )
                    continue

                confidence = float(detected_box.conf[0])
                threshold = _confidence_threshold_for_class(class_name)
                if confidence < threshold:
                    logger.info(
                        "detection rejected low_confidence rawClassId=%s mappedLabel=%s confidence=%.4f threshold=%.4f modelPath=%s",
                        class_id,
                        class_name,
                        confidence,
                        threshold,
                        self.model_path,
                    )
                    continue

                raw_bbox = tuple(float(value) for value in detected_box.xyxy[0].tolist())
                bbox_validation = _validate_and_clamp_bbox(raw_bbox, frame_width, frame_height)
                if not bbox_validation["ok"]:
                    logger.info(
                        "detection rejected bbox reason=%s rawClassId=%s mappedLabel=%s confidence=%.4f bbox=%s areaRatio=%.5f outsideRatio=%.5f",
                        bbox_validation["reason"],
                        class_id,
                        class_name,
                        confidence,
                        tuple(round(value, 1) for value in raw_bbox),
                        bbox_validation["areaRatio"],
                        bbox_validation["outsideRatio"],
                    )
                    continue

                bbox = bbox_validation["bbox"]
                x1, y1, x2, y2 = bbox
                track = self._assign_track(class_name, bbox, confidence, now)
                track_id = track.track_id
                if track.consecutive_frames < MIN_CONSECUTIVE_FRAMES:
                    logger.info(
                        "detection pending confirmation frameClass=%s confidence=%.4f threshold=%.4f trackId=%s consecutiveFrames=%s/%s bboxAreaRatio=%.5f frameQuality=%s",
                        class_name,
                        confidence,
                        threshold,
                        track_id,
                        track.consecutive_frames,
                        MIN_CONSECUTIVE_FRAMES,
                        bbox_validation["areaRatio"],
                        frame_quality["reason"],
                    )
                    continue

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
                        "confidenceThreshold": threshold,
                        "confirmed": True,
                        "consecutiveFrames": track.consecutive_frames,
                        "firstSeenEpochMillis": int((time.time() - max(0.0, now - track.first_seen)) * 1000),
                        "maxConfidence": track.max_confidence,
                        "averageConfidence": track.confidence_sum / max(1, track.observation_count),
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
                        "bboxAreaRatio": bbox_validation["areaRatio"],
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
                "frameQuality": frame_quality,
                "confirmation": {
                    "minConsecutiveFrames": MIN_CONSECUTIVE_FRAMES,
                    "classConfidenceThresholds": CLASS_CONFIDENCE_THRESHOLDS,
                },
                "calibrationWarning": self.runtime_focal_warning,
                "calibrationPath": self.config.calibration_path,
                "runtimeFocalLengthPixels": self.runtime_focal_length,
            }

    def _assign_track(self, class_name: str, bbox: tuple[float, float, float, float], confidence: float, now: float) -> _TrackMemory:
        best_track = None
        best_overlap = 0.0
        for track in self.web_tracks:
            if track.class_name != class_name:
                continue
            overlap = _iou(track.bbox, bbox)
            if overlap > best_overlap:
                best_track = track
                best_overlap = overlap
        if best_track is not None and best_overlap >= 0.20:
            best_track.bbox = bbox
            best_track.last_seen = now
            best_track.consecutive_frames += 1
            best_track.max_confidence = max(best_track.max_confidence, confidence)
            best_track.confidence_sum += confidence
            best_track.observation_count += 1
            return best_track

        track_id = self.next_track_id
        self.next_track_id += 1
        track = _TrackMemory(
            track_id=track_id,
            class_name=class_name,
            bbox=bbox,
            last_seen=now,
            first_seen=now,
            consecutive_frames=1,
            max_confidence=confidence,
            confidence_sum=confidence,
            observation_count=1,
        )
        self.web_tracks.append(track)
        return track

    def _cleanup_web_tracks(self, now: float) -> None:
        self.web_tracks = [track for track in self.web_tracks if now - track.last_seen <= TRACK_STALE_SECONDS]

    def _reset_transient_state(self) -> None:
        self.web_tracks.clear()
        self.distance_history = DistanceHistory()
        self.motion_analyzer = RelativeMotionAnalyzer()
        self.risk_smoother = RiskSmoother()
        self.track_history = TrackHistory()
        self.warning_manager = WarningManager(warnings_enabled=True, audio_enabled=False)

    def _empty_result(self, frame_width: int, frame_height: int, frame_quality: dict[str, Any]) -> dict[str, Any]:
        timestamp_ms = int(time.time() * 1000)
        return {
            "ok": True,
            "modelPath": str(self.model_path) if self.model_path else None,
            "usingPretrainedFallback": self.using_pretrained_fallback,
            "frameWidth": frame_width,
            "frameHeight": frame_height,
            "sceneRisk": "UNKNOWN",
            "warning": {
                "level": NONE,
                "message": "",
                "trackId": None,
                "className": None,
                "riskLevel": "UNKNOWN",
                "distanceMeters": None,
                "ttcSeconds": None,
            },
            "detections": [],
            "timestamp": timestamp_ms,
            "frameQuality": frame_quality,
            "confirmation": {
                "minConsecutiveFrames": MIN_CONSECUTIVE_FRAMES,
                "classConfidenceThresholds": CLASS_CONFIDENCE_THRESHOLDS,
            },
            "calibrationWarning": self.runtime_focal_warning,
            "calibrationPath": self.config.calibration_path,
            "runtimeFocalLengthPixels": self.runtime_focal_length,
            }


def _confidence_threshold_for_class(class_name: str) -> float:
    return CLASS_CONFIDENCE_THRESHOLDS.get(class_name.lower(), DEFAULT_CLASS_CONFIDENCE)


def _analyze_frame_quality(frame: Any) -> dict[str, Any]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 60, 160)
    edge_ratio = float(np.count_nonzero(edges)) / float(max(1, gray.size))
    reason = "GOOD"
    ok = True
    if brightness < QUALITY_MIN_BRIGHTNESS:
        ok = False
        reason = "TOO_DARK"
    elif brightness > QUALITY_MAX_BRIGHTNESS:
        ok = False
        reason = "OVEREXPOSED"
    elif contrast < QUALITY_MIN_CONTRAST:
        ok = False
        reason = "LOW_VISUAL_INFORMATION"
    elif blur_variance < QUALITY_MIN_BLUR_VARIANCE and edge_ratio < QUALITY_MIN_EDGE_RATIO:
        ok = False
        reason = "BLUR_OR_OBSTRUCTION"
    elif edge_ratio < QUALITY_MIN_EDGE_RATIO:
        ok = False
        reason = "CAMERA_OBSTRUCTED"

    return {
        "ok": ok,
        "reason": reason,
        "brightness": brightness,
        "contrast": contrast,
        "blurVariance": blur_variance,
        "edgeRatio": edge_ratio,
    }


def _validate_and_clamp_bbox(
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    left, top, right, bottom = bbox
    if not all(np.isfinite(value) for value in bbox):
        return _bbox_result(False, "NON_FINITE_COORDINATES", bbox, bbox, frame_width, frame_height)
    raw_width = right - left
    raw_height = bottom - top
    if raw_width <= 0 or raw_height <= 0:
        return _bbox_result(False, "INVALID_COORDINATES", bbox, bbox, frame_width, frame_height)

    clamped = (
        float(np.clip(left, 0, frame_width)),
        float(np.clip(top, 0, frame_height)),
        float(np.clip(right, 0, frame_width)),
        float(np.clip(bottom, 0, frame_height)),
    )
    c_left, c_top, c_right, c_bottom = clamped
    width = max(0.0, c_right - c_left)
    height = max(0.0, c_bottom - c_top)
    if width < MIN_BOX_DIMENSION_PX or height < MIN_BOX_DIMENSION_PX:
        return _bbox_result(False, "BOX_TOO_SMALL", bbox, clamped, frame_width, frame_height)

    frame_area = float(max(1, frame_width * frame_height))
    raw_area = max(0.0, raw_width * raw_height)
    clamped_area = width * height
    area_ratio = clamped_area / frame_area
    outside_ratio = max(0.0, raw_area - clamped_area) / max(1.0, raw_area)
    if area_ratio < MIN_BOX_AREA_RATIO:
        return _bbox_result(False, "BOX_AREA_TOO_SMALL", bbox, clamped, frame_width, frame_height)
    if area_ratio > MAX_BOX_AREA_RATIO:
        return _bbox_result(False, "BOX_AREA_TOO_LARGE", bbox, clamped, frame_width, frame_height)
    if outside_ratio > MAX_OUTSIDE_RATIO:
        return _bbox_result(False, "BOX_MOSTLY_OUTSIDE_FRAME", bbox, clamped, frame_width, frame_height)

    return _bbox_result(True, "OK", bbox, clamped, frame_width, frame_height)


def _bbox_result(
    ok: bool,
    reason: str,
    raw_bbox: tuple[float, float, float, float],
    clamped_bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    left, top, right, bottom = raw_bbox
    c_left, c_top, c_right, c_bottom = clamped_bbox
    raw_area = max(0.0, right - left) * max(0.0, bottom - top)
    clamped_area = max(0.0, c_right - c_left) * max(0.0, c_bottom - c_top)
    frame_area = float(max(1, frame_width * frame_height))
    return {
        "ok": ok,
        "reason": reason,
        "bbox": clamped_bbox,
        "areaRatio": clamped_area / frame_area,
        "outsideRatio": max(0.0, raw_area - clamped_area) / max(1.0, raw_area),
    }


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
