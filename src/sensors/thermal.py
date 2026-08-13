"""Generic thermal-camera placeholder for future RGB/thermal fusion."""

from __future__ import annotations

from src.sensors.base import NOT_CONFIGURED, SensorMeasurement, SensorSource, SensorStatus


class ThermalSensorSource(SensorSource):
    sensor_type = "THERMAL"

    def __init__(self, source: str | None = None) -> None:
        self.source = source

    def get_latest_measurements(self, now: float | None = None) -> list[SensorMeasurement]:
        return []

    def status(self, now: float | None = None) -> SensorStatus:
        if not self.source:
            return SensorStatus(self.sensor_type, NOT_CONFIGURED, "thermal camera not configured")
        return SensorStatus(self.sensor_type, NOT_CONFIGURED, "thermal-specific detection model and alignment pending")
