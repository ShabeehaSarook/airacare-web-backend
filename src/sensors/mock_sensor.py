"""Mock radar-like sensor source for safe Phase 15 development."""

from __future__ import annotations

import csv
from pathlib import Path

from src.sensors.base import MOCK, SensorMeasurement, SensorSource, SensorStatus, current_time


class MockSensorSource(SensorSource):
    """Return simulated radar-style range and relative speed measurements.

    CSV replay uses relative timestamps in seconds:
    timestamp,distance_m,relative_speed_mps,angle_deg
    """

    sensor_type = "MOCK_RADAR"

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else None
        self.rows: list[SensorMeasurement] = []
        self.started_at: float | None = None
        self.message = "generated mock radar target"
        if self.data_path:
            self._load_csv(self.data_path)

    def _load_csv(self, path: Path) -> None:
        if not path.exists():
            self.message = f"mock CSV not found: {path}"
            return
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader, start=1):
                try:
                    timestamp = float(row["timestamp"])
                    distance = _optional_float(row.get("distance_m"))
                    speed = _optional_float(row.get("relative_speed_mps"))
                    angle = _optional_float(row.get("angle_deg"))
                except (KeyError, TypeError, ValueError):
                    continue
                self.rows.append(
                    SensorMeasurement(
                        sensor_type=self.sensor_type,
                        timestamp=timestamp,
                        distance_m=distance,
                        relative_speed_mps=speed,
                        angle_deg=angle,
                        confidence=0.90,
                        sensor_object_id=f"mock-{index}",
                    )
                )
        self.message = f"loaded {len(self.rows)} mock measurements from {path}"

    def start(self) -> None:
        self.started_at = current_time()

    def get_latest_measurements(self, now: float | None = None) -> list[SensorMeasurement]:
        timestamp = current_time() if now is None else now
        if self.started_at is None:
            self.start()

        elapsed = timestamp - float(self.started_at)
        if self.rows:
            row = self._row_for_elapsed(elapsed)
            return [
                SensorMeasurement(
                    sensor_type=self.sensor_type,
                    timestamp=timestamp,
                    distance_m=row.distance_m,
                    relative_speed_mps=row.relative_speed_mps,
                    angle_deg=row.angle_deg,
                    confidence=row.confidence,
                    sensor_object_id=row.sensor_object_id,
                )
            ]

        distance = max(2.0, 20.0 - 5.0 * elapsed)
        return [
            SensorMeasurement(
                sensor_type=self.sensor_type,
                timestamp=timestamp,
                distance_m=distance,
                relative_speed_mps=5.0,
                angle_deg=0.0,
                confidence=0.90,
                sensor_object_id="mock-1",
            )
        ]

    def _row_for_elapsed(self, elapsed: float) -> SensorMeasurement:
        if not self.rows:
            raise RuntimeError("No mock sensor rows loaded")
        duration = max(self.rows[-1].timestamp, 0.001)
        replay_time = elapsed % duration
        selected = self.rows[0]
        for row in self.rows:
            if row.timestamp <= replay_time:
                selected = row
            else:
                break
        return selected

    def status(self, now: float | None = None) -> SensorStatus:
        return SensorStatus(self.sensor_type, MOCK, self.message)


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
