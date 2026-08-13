"""Lightweight image enhancement for Phase 14 low-visibility frames."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.visibility import DAY, LOW_LIGHT, VERY_DARK, DEFAULT_VISIBILITY_CONFIG


@dataclass
class EnhancementResult:
    frame: object
    applied: bool
    gamma: float | None
    clahe_enabled: bool
    denoise_enabled: bool
    sharpen_enabled: bool
    elapsed_ms: float
    mode_used: str


def should_enhance(mode: str, visibility_class: str) -> bool:
    mode = mode.lower()
    if mode == "off":
        return False
    if mode == "lowlight":
        return True
    if mode == "auto":
        return visibility_class in {LOW_LIGHT, VERY_DARK}
    return False


def enhance_frame(frame, visibility_class: str, mode: str = "auto", config: dict | None = None) -> EnhancementResult:
    cfg = DEFAULT_VISIBILITY_CONFIG if config is None else config
    start = time.perf_counter()

    if not should_enhance(mode, visibility_class):
        return EnhancementResult(frame, False, None, False, False, False, 0.0, "off")

    gamma = float(cfg["gamma_very_dark"] if visibility_class == VERY_DARK else cfg["gamma_low_light"])
    enhanced = _apply_luminance_clahe(frame, cfg) if bool(cfg.get("clahe_enabled", True)) else frame.copy()
    enhanced = _apply_gamma(enhanced, gamma)

    denoise_enabled = bool(cfg.get("denoise_enabled", False))
    if denoise_enabled:
        h = float(cfg.get("denoise_h", 4))
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, h, h, 7, 21)

    sharpen_enabled = bool(cfg.get("sharpen_enabled", False))
    if sharpen_enabled:
        enhanced = _mild_sharpen(enhanced)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return EnhancementResult(
        enhanced,
        True,
        gamma,
        bool(cfg.get("clahe_enabled", True)),
        denoise_enabled,
        sharpen_enabled,
        elapsed_ms,
        mode,
    )


def _apply_luminance_clahe(frame, config: dict):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    grid_size = int(config.get("clahe_grid_size", 8))
    clahe = cv2.createCLAHE(
        clipLimit=float(config.get("clahe_clip_limit", 2.0)),
        tileGridSize=(grid_size, grid_size),
    )
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _apply_gamma(frame, gamma: float):
    inv_gamma = 1.0 / max(gamma, 0.01)
    table = np.array([((value / 255.0) ** inv_gamma) * 255 for value in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, table)


def _mild_sharpen(frame):
    blurred = cv2.GaussianBlur(frame, (0, 0), 1.0)
    return cv2.addWeighted(frame, 1.25, blurred, -0.25, 0)
