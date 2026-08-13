"""Monocular distance-estimation helpers for Phase 09.

Distances are estimates based on camera calibration and representative
object-size assumptions. They are not radar-grade or guaranteed physical
measurements.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

from src.config import KNOWN_OBJECT_HEIGHTS_METERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = PROJECT_ROOT / "config" / "camera_calibration.json"
DISTANCE_HISTORY_LIMIT = 5
MIN_BBOX_HEIGHT_PIXELS = 4.0
STALE_DISTANCE_SECONDS = 2.0


def load_calibration(path: str | Path) -> dict[str, Any] | None:
    """Load a camera calibration JSON file."""
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None

    with calibration_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    focal_length = data.get("focal_length_pixels")
    if not isinstance(focal_length, (int, float)) or focal_length <= 0:
        raise ValueError("Calibration file must contain a positive focal_length_pixels value.")

    return data


def get_calibration_resolution(calibration: dict[str, Any]) -> tuple[int, int] | None:
    """Return calibration resolution as (width, height), if present.

    Newer calibration files use ``calibration_resolution``. The loader also
    accepts ``calibration_width`` and ``calibration_height`` so older/manual
    files still work.
    """
    resolution = calibration.get("calibration_resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        try:
            width = int(resolution[0])
            height = int(resolution[1])
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0:
            return width, height

    try:
        width = int(calibration.get("calibration_width"))
        height = int(calibration.get("calibration_height"))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def focal_length_for_runtime_resolution(
    calibration: dict[str, Any],
    runtime_resolution: tuple[int, int],
) -> tuple[float, str | None]:
    """Scale focal length when runtime frame height differs from calibration height."""
    focal_length = float(calibration["focal_length_pixels"])
    calibration_resolution = get_calibration_resolution(calibration)
    runtime_width, runtime_height = runtime_resolution

    if calibration_resolution is None:
        return focal_length, "Calibration resolution is missing; distance accuracy may be reduced."

    calibration_width, calibration_height = calibration_resolution
    if calibration_width == runtime_width and calibration_height == runtime_height:
        return focal_length, None

    height_scale = runtime_height / calibration_height
    width_scale = runtime_width / calibration_width
    scaled_focal_length = focal_length * height_scale

    calibration_aspect = calibration_width / calibration_height
    runtime_aspect = runtime_width / runtime_height
    aspect_delta = abs(calibration_aspect - runtime_aspect) / calibration_aspect
    if aspect_delta > 0.05:
        return (
            scaled_focal_length,
            "Camera aspect ratio differs from calibration; distance accuracy may be reduced.",
        )

    if abs(width_scale - height_scale) > 0.05:
        return (
            scaled_focal_length,
            "Camera resolution scaling is not uniform; distance accuracy may be reduced.",
        )

    return scaled_focal_length, "Runtime resolution differs from calibration; focal length was scaled."


def estimate_distance(
    class_name: str,
    bbox_height_pixels: float,
    focal_length_pixels: float,
) -> float | None:
    """Estimate distance in metres using the pinhole-camera approximation."""
    if bbox_height_pixels < MIN_BBOX_HEIGHT_PIXELS:
        return None
    if focal_length_pixels <= 0:
        return None

    known_height = KNOWN_OBJECT_HEIGHTS_METERS.get(class_name.lower())
    if known_height is None or known_height <= 0:
        return None

    return (known_height * focal_length_pixels) / bbox_height_pixels


class DistanceHistory:
    """Keep short per-track distance histories for light median smoothing."""

    def __init__(
        self,
        history_limit: int = DISTANCE_HISTORY_LIMIT,
        stale_seconds: float = STALE_DISTANCE_SECONDS,
    ) -> None:
        self.history_limit = history_limit
        self.stale_seconds = stale_seconds
        self.distances: dict[int, deque[float]] = {}
        self.last_seen: dict[int, float] = {}

    def update(self, track_id: int, distance_meters: float | None) -> float | None:
        if distance_meters is None:
            return None

        history = self.distances.setdefault(track_id, deque(maxlen=self.history_limit))
        history.append(distance_meters)
        self.last_seen[track_id] = time.perf_counter()
        return float(median(history))

    def cleanup_stale(self) -> None:
        now = time.perf_counter()
        stale_ids = [
            track_id
            for track_id, last_seen_time in self.last_seen.items()
            if now - last_seen_time > self.stale_seconds
        ]
        for track_id in stale_ids:
            self.last_seen.pop(track_id, None)
            self.distances.pop(track_id, None)

