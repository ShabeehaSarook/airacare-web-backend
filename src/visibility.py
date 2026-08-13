"""Frame visibility analysis for Phase 14."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISIBILITY_CONFIG_PATH = PROJECT_ROOT / "config" / "visibility_config.json"
DAY = "DAY"
LOW_LIGHT = "LOW_LIGHT"
VERY_DARK = "VERY_DARK"

DEFAULT_VISIBILITY_CONFIG = {
    "brightness_low_threshold": 85.0,
    "brightness_very_dark_threshold": 35.0,
    "contrast_low_threshold": 28.0,
    "blur_low_threshold": 60.0,
    "gamma_low_light": 1.45,
    "gamma_very_dark": 1.85,
    "clahe_enabled": True,
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": 8,
    "denoise_enabled": False,
    "denoise_h": 4,
    "sharpen_enabled": False,
    "highlight_clip_percentile": 99.5,
    "max_enhancement_time_warning_ms": 25.0,
}


@dataclass
class VisibilityMetrics:
    brightness: float
    contrast: float
    blur_score: float
    visibility_class: str


def load_visibility_config(path: str | Path = DEFAULT_VISIBILITY_CONFIG_PATH) -> dict:
    config = dict(DEFAULT_VISIBILITY_CONFIG)
    config_path = Path(path)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8-sig") as file:
            user_config = json.load(file)
        config.update(user_config)
    return config


def analyze_visibility(frame, config: dict | None = None) -> VisibilityMetrics:
    cfg = DEFAULT_VISIBILITY_CONFIG if config is None else config
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness <= float(cfg["brightness_very_dark_threshold"]):
        visibility_class = VERY_DARK
    elif brightness <= float(cfg["brightness_low_threshold"]) or contrast <= float(cfg["contrast_low_threshold"]):
        visibility_class = LOW_LIGHT
    else:
        visibility_class = DAY

    return VisibilityMetrics(
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        visibility_class=visibility_class,
    )

