"""Phase 15 sensor association and fusion.

The fusion layer enriches camera YOLO tracks with optional sensor measurements.
It does not fabricate sensor data and falls back to camera-only estimates when
association is stale, ambiguous, or unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.relative_motion import CLOSING_THRESHOLD_MPS
from src.sensors.base import SensorMeasurement


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENSOR_CALIBRATION_PATH = PROJECT_ROOT / "config" / "sensor_calibration.json"

DEFAULT_SENSOR_FUSION_CONFIG = {
    "camera_horizontal_fov_deg": 70.0,
    "max_sensor_time_difference_seconds": 0.25,
    "max_angle_difference_deg": 12.0,
    "max_distance_difference_m": 10.0,
    "distance_priority": ["RADAR", "DEPTH", "MOCK_RADAR", "CAMERA"],
    "closing_speed_priority": ["RADAR", "MOCK_RADAR", "CAMERA"],
    "sensor_stale_timeout_seconds": 0.5,
}


@dataclass
class CameraObjectState:
    track_id: int
    class_name: str
    detection_confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    distance_camera_m: float | None
    camera_closing_speed_mps: float | None
    movement: str
    path_zone: str
    timestamp: float


@dataclass
class FusedObjectState:
    track_id: int
    class_name: str
    class_confidence: float
    camera_distance_m: float | None
    sensor_distance_m: float | None
    fused_distance_m: float | None
    camera_closing_speed_mps: float | None
    sensor_relative_speed_mps: float | None
    fused_closing_speed_mps: float | None
    path_zone: str
    sensor_sources: list[str] = field(default_factory=list)
    distance_source: str = "CAMERA"
    closing_speed_source: str = "CAMERA"
    fusion_confidence: str = "CAMERA_ONLY"
    timestamp: float = 0.0
    ttc_seconds: float | None = None
    sensor_object_id: str | int | None = None
    association_status: str = "NO_SENSOR"
    camera_bearing_deg: float | None = None
    sensor_angle_deg: float | None = None
    timestamp_difference_s: float | None = None


def load_sensor_fusion_config(path: str | Path | None = None) -> dict:
    config = dict(DEFAULT_SENSOR_FUSION_CONFIG)
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        return config

    with config_path.open("r", encoding="utf-8-sig") as file:
        loaded = json.load(file)
    if isinstance(loaded, dict):
        config.update(loaded)
    return config


def camera_x_to_bearing_deg(center_x: float, frame_width: int, horizontal_fov_deg: float) -> float:
    if frame_width <= 0:
        return 0.0
    normalized = (center_x / frame_width) - 0.5
    return normalized * horizontal_fov_deg


class SensorFusionEngine:
    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(DEFAULT_SENSOR_FUSION_CONFIG)
        if config:
            self.config.update(config)

    def fuse(
        self,
        camera_object: CameraObjectState,
        sensor_measurements: list[SensorMeasurement],
        frame_width: int,
    ) -> FusedObjectState:
        camera_bearing = camera_x_to_bearing_deg(
            camera_object.center[0],
            frame_width,
            float(self.config["camera_horizontal_fov_deg"]),
        )
        association = self._associate(camera_object, sensor_measurements, camera_bearing)
        if association is None:
            return self._camera_only(camera_object, camera_bearing, "NO_MATCH" if sensor_measurements else "NO_SENSOR")

        measurement, confidence, timestamp_diff = association
        source = measurement.sensor_type
        sensor_distance = measurement.distance_m if _valid_positive(measurement.distance_m) else None
        sensor_speed = measurement.relative_speed_mps

        fused_distance, distance_source = self._choose_distance(camera_object.distance_camera_m, sensor_distance, source)
        fused_speed, speed_source = self._choose_speed(camera_object.camera_closing_speed_mps, sensor_speed, source)
        ttc = fused_distance / fused_speed if fused_distance is not None and fused_speed is not None and fused_speed > CLOSING_THRESHOLD_MPS else None

        return FusedObjectState(
            track_id=camera_object.track_id,
            class_name=camera_object.class_name,
            class_confidence=camera_object.detection_confidence,
            camera_distance_m=camera_object.distance_camera_m,
            sensor_distance_m=sensor_distance,
            fused_distance_m=fused_distance,
            camera_closing_speed_mps=camera_object.camera_closing_speed_mps,
            sensor_relative_speed_mps=sensor_speed,
            fused_closing_speed_mps=fused_speed,
            path_zone=camera_object.path_zone,
            sensor_sources=[source],
            distance_source=distance_source,
            closing_speed_source=speed_source,
            fusion_confidence=confidence,
            timestamp=camera_object.timestamp,
            ttc_seconds=ttc,
            sensor_object_id=measurement.sensor_object_id,
            association_status="ASSOCIATED",
            camera_bearing_deg=camera_bearing,
            sensor_angle_deg=measurement.angle_deg,
            timestamp_difference_s=timestamp_diff,
        )

    def _camera_only(self, camera_object: CameraObjectState, camera_bearing: float, status: str) -> FusedObjectState:
        distance = camera_object.distance_camera_m
        speed = camera_object.camera_closing_speed_mps
        ttc = distance / speed if distance is not None and speed is not None and speed > CLOSING_THRESHOLD_MPS else None
        return FusedObjectState(
            track_id=camera_object.track_id,
            class_name=camera_object.class_name,
            class_confidence=camera_object.detection_confidence,
            camera_distance_m=distance,
            sensor_distance_m=None,
            fused_distance_m=distance,
            camera_closing_speed_mps=speed,
            sensor_relative_speed_mps=None,
            fused_closing_speed_mps=speed,
            path_zone=camera_object.path_zone,
            distance_source="CAMERA" if distance is not None else "UNKNOWN",
            closing_speed_source="CAMERA" if speed is not None else "UNKNOWN",
            fusion_confidence="CAMERA_ONLY",
            timestamp=camera_object.timestamp,
            ttc_seconds=ttc,
            association_status=status,
            camera_bearing_deg=camera_bearing,
        )

    def _associate(
        self,
        camera_object: CameraObjectState,
        measurements: list[SensorMeasurement],
        camera_bearing: float,
    ) -> tuple[SensorMeasurement, str, float] | None:
        candidates: list[tuple[float, SensorMeasurement, float]] = []
        max_time = float(self.config["max_sensor_time_difference_seconds"])
        max_angle = float(self.config["max_angle_difference_deg"])
        max_distance = float(self.config["max_distance_difference_m"])

        for measurement in measurements:
            timestamp_diff = abs(camera_object.timestamp - measurement.timestamp)
            if timestamp_diff > max_time:
                continue

            score = 100.0 - (timestamp_diff / max_time) * 25.0
            if measurement.angle_deg is not None:
                angle_diff = abs(camera_bearing - measurement.angle_deg)
                if angle_diff > max_angle:
                    continue
                score -= (angle_diff / max_angle) * 45.0

            if camera_object.distance_camera_m is not None and measurement.distance_m is not None:
                distance_diff = abs(camera_object.distance_camera_m - measurement.distance_m)
                if distance_diff > max_distance:
                    score -= 25.0
                else:
                    score -= (distance_diff / max_distance) * 20.0

            candidates.append((score, measurement, timestamp_diff))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 8.0:
            return None

        score, measurement, timestamp_diff = candidates[0]
        if score >= 70.0:
            confidence = "HIGH"
        elif score >= 45.0:
            confidence = "MEDIUM"
        else:
            return None
        return measurement, confidence, timestamp_diff

    def _choose_distance(self, camera_distance: float | None, sensor_distance: float | None, sensor_source: str) -> tuple[float | None, str]:
        for source in self.config["distance_priority"]:
            if source == sensor_source and sensor_distance is not None:
                return sensor_distance, sensor_source
            if source == "CAMERA" and camera_distance is not None:
                return camera_distance, "CAMERA"
        return None, "UNKNOWN"

    def _choose_speed(self, camera_speed: float | None, sensor_speed: float | None, sensor_source: str) -> tuple[float | None, str]:
        for source in self.config["closing_speed_priority"]:
            if source == sensor_source and sensor_speed is not None:
                return sensor_speed, sensor_source
            if source == "CAMERA" and camera_speed is not None:
                return camera_speed, "CAMERA"
        return None, "UNKNOWN"


def _valid_positive(value: float | None) -> bool:
    return value is not None and value > 0
