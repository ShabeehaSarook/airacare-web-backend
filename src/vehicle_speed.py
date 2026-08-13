"""Vehicle speed providers for Phase 11.

Manual speed is used now. GPS can be added later by implementing the same
get_vehicle_speed_kmh() interface with live GPS data.
"""

from __future__ import annotations

from dataclasses import dataclass


def kmh_to_mps(speed_kmh: float) -> float:
    """Convert kilometres per hour to metres per second."""
    return speed_kmh / 3.6


class VehicleSpeedProvider:
    """Base interface for current vehicle speed sources."""

    source_name = "unknown"

    def get_vehicle_speed_kmh(self) -> float:
        raise NotImplementedError

    def get_vehicle_speed_mps(self) -> float:
        return kmh_to_mps(self.get_vehicle_speed_kmh())


@dataclass
class ManualSpeedProvider(VehicleSpeedProvider):
    """Development/test vehicle speed supplied by CLI."""

    speed_kmh: float = 0.0
    source_name: str = "manual"

    def get_vehicle_speed_kmh(self) -> float:
        return max(0.0, float(self.speed_kmh))


class GPSSpeedProvider(VehicleSpeedProvider):
    """Placeholder interface for future GPS-derived speed."""

    source_name = "gps"

    def __init__(self, speed_kmh: float | None = None, speed_mps: float | None = None) -> None:
        self.speed_kmh = speed_kmh
        self.speed_mps = speed_mps

    def get_vehicle_speed_kmh(self) -> float:
        if self.speed_kmh is not None:
            return max(0.0, float(self.speed_kmh))
        if self.speed_mps is not None:
            return max(0.0, float(self.speed_mps) * 3.6)
        return 0.0
