"""Sensor abstractions for Phase 15 optional sensor fusion.

External sensors are optional. The RGB camera remains the primary source for
class labels and visual tracking; sensors can enrich distance and speed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


CONNECTED = "CONNECTED"
DISCONNECTED = "DISCONNECTED"
STALE = "STALE"
INVALID = "INVALID"
MOCK = "MOCK"
NOT_CONFIGURED = "NOT_CONFIGURED"


@dataclass
class SensorMeasurement:
    sensor_type: str
    timestamp: float
    distance_m: float | None = None
    relative_speed_mps: float | None = None
    angle_deg: float | None = None
    confidence: float | None = None
    sensor_object_id: str | int | None = None


@dataclass
class SensorStatus:
    sensor_type: str
    state: str
    message: str = ""


class SensorSource:
    """Base class for non-blocking latest-measurement sensor sources."""

    sensor_type = "SENSOR"

    def start(self) -> None:
        """Start the sensor source if it needs a background reader."""

    def stop(self) -> None:
        """Stop the sensor source and release resources."""

    def get_latest_measurements(self, now: float | None = None) -> list[SensorMeasurement]:
        """Return recent measurements without blocking the video loop."""
        return []

    def status(self, now: float | None = None) -> SensorStatus:
        return SensorStatus(self.sensor_type, NOT_CONFIGURED)


def current_time() -> float:
    return time.perf_counter()
