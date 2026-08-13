"""Generic depth-sensor placeholder for future hardware-specific integration."""

from __future__ import annotations

from src.sensors.base import NOT_CONFIGURED, SensorMeasurement, SensorSource, SensorStatus


class DepthSensorSource(SensorSource):
    sensor_type = "DEPTH"

    def get_latest_measurements(self, now: float | None = None) -> list[SensorMeasurement]:
        return []

    def status(self, now: float | None = None) -> SensorStatus:
        return SensorStatus(self.sensor_type, NOT_CONFIGURED, "depth sensor integration pending")
