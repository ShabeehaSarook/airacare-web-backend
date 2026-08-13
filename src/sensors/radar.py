"""Generic radar placeholder for future hardware-specific integration."""

from __future__ import annotations

from src.sensors.base import DISCONNECTED, NOT_CONFIGURED, SensorMeasurement, SensorSource, SensorStatus


class RadarSensorSource(SensorSource):
    sensor_type = "RADAR"

    def __init__(self, port: str | None = None, baud: int = 115200) -> None:
        self.port = port
        self.baud = baud

    def get_latest_measurements(self, now: float | None = None) -> list[SensorMeasurement]:
        return []

    def status(self, now: float | None = None) -> SensorStatus:
        if not self.port:
            return SensorStatus(self.sensor_type, NOT_CONFIGURED, "radar hardware integration pending")
        return SensorStatus(self.sensor_type, DISCONNECTED, f"no radar reader implemented for {self.port} at {self.baud}")
