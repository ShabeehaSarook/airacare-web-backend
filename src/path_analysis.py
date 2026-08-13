"""Phase 10-compatible path classification helper.

If a path config file is unavailable, a simple image-based default is used:
center-bottom area is IN_PATH, the wider surrounding band is NEAR_PATH, and the
rest is OUTSIDE_PATH. This is only geometric context, not a risk warning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH_CONFIG_PATH = PROJECT_ROOT / "config" / "path_config.json"
PATH_IN = "IN_PATH"
PATH_NEAR = "NEAR_PATH"
PATH_OUTSIDE = "OUTSIDE_PATH"


@dataclass
class PathAnalyzer:
    in_path_polygon: list[tuple[float, float]] | None = None
    near_path_polygon: list[tuple[float, float]] | None = None

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_PATH_CONFIG_PATH) -> "PathAnalyzer":
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(
            in_path_polygon=_parse_polygon(data.get("in_path_polygon")),
            near_path_polygon=_parse_polygon(data.get("near_path_polygon")),
        )

    def classify(self, center: tuple[float, float], frame_width: int, frame_height: int) -> str:
        if self.in_path_polygon and _point_in_polygon(center, self.in_path_polygon, frame_width, frame_height):
            return PATH_IN
        if self.near_path_polygon and _point_in_polygon(center, self.near_path_polygon, frame_width, frame_height):
            return PATH_NEAR
        return self._default_classify(center, frame_width, frame_height)

    def _default_classify(self, center: tuple[float, float], frame_width: int, frame_height: int) -> str:
        x, y = center
        x_ratio = x / frame_width if frame_width else 0.0
        y_ratio = y / frame_height if frame_height else 0.0
        if y_ratio >= 0.45 and 0.35 <= x_ratio <= 0.65:
            return PATH_IN
        if y_ratio >= 0.35 and 0.18 <= x_ratio <= 0.82:
            return PATH_NEAR
        return PATH_OUTSIDE


def _parse_polygon(raw_polygon: Any) -> list[tuple[float, float]] | None:
    if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
        return None
    polygon = []
    for point in raw_polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        polygon.append((float(point[0]), float(point[1])))
    return polygon


def _point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> bool:
    scaled_polygon = []
    for x, y in polygon:
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            scaled_polygon.append((x * frame_width, y * frame_height))
        else:
            scaled_polygon.append((x, y))
    contour = np.array(scaled_polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0
