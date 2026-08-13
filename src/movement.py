"""Image-plane movement direction helpers for Phase 08 tracking."""

from __future__ import annotations


MOVEMENT_THRESHOLD_PIXELS = 8


def calculate_direction(
    previous_center: tuple[float, float] | None,
    current_center: tuple[float, float],
    threshold_pixels: float = MOVEMENT_THRESHOLD_PIXELS,
) -> str:
    """Return a simple image-plane direction label from two center points."""
    if previous_center is None:
        return "STATIONARY"

    delta_x = current_center[0] - previous_center[0]
    delta_y = current_center[1] - previous_center[1]

    horizontal = ""
    vertical = ""

    if abs(delta_x) >= threshold_pixels:
        horizontal = "RIGHT" if delta_x > 0 else "LEFT"

    if abs(delta_y) >= threshold_pixels:
        vertical = "DOWN" if delta_y > 0 else "UP"

    if vertical and horizontal:
        return f"{vertical}-{horizontal}"
    if horizontal:
        return horizontal
    if vertical:
        return vertical
    return "STATIONARY"
