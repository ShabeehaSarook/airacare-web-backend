"""Driver warning manager for Phase 13.

Warnings are generated from Phase 12 risk results. This module does not
implement automatic braking or vehicle control.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.audio_alert import AudioAlertPlayer
from src.config import WARNING_CONFIG
from src.risk import HIGH_RISK, LOW_RISK, MEDIUM_RISK, RISK_ORDER, UNKNOWN


NONE = "NONE"
CAUTION = "CAUTION"
DANGER = "DANGER"
RISK_TO_WARNING = {
    UNKNOWN: NONE,
    LOW_RISK: NONE,
    MEDIUM_RISK: CAUTION,
    HIGH_RISK: DANGER,
}
WARNING_ORDER = {
    NONE: 0,
    CAUTION: 1,
    DANGER: 2,
}


@dataclass
class WarningCandidate:
    track_id: int
    class_name: str
    risk_level: str
    zone: str
    distance_m: float | None
    ttc_seconds: float | None
    reason: str = ""


@dataclass
class WarningState:
    level: str = NONE
    track_id: int | None = None
    class_name: str | None = None
    message: str = ""
    zone: str | None = None
    distance_m: float | None = None
    ttc_seconds: float | None = None
    audio_triggered: bool = False
    cooldown_remaining_seconds: float = 0.0
    previous_level: str = NONE
    risk_level: str = UNKNOWN
    reason: str = ""


class WarningManager:
    """Select the highest-priority threat and throttle audio warnings."""

    def __init__(self, warnings_enabled: bool = True, audio_enabled: bool = True, config: dict | None = None) -> None:
        self.config = WARNING_CONFIG if config is None else config
        self.warnings_enabled = warnings_enabled
        self.audio_player = AudioAlertPlayer(enabled=audio_enabled)
        self.previous_level = NONE
        self.last_audio_time_by_level = {CAUTION: 0.0, DANGER: 0.0}
        self.last_warning_by_track: dict[int, str] = {}
        self.last_seen_by_track: dict[int, float] = {}

    def update(self, candidates: list[WarningCandidate], timestamp: float | None = None) -> WarningState:
        now = time.perf_counter() if timestamp is None else timestamp
        self._cleanup_stale(now)

        if not self.warnings_enabled:
            self.previous_level = NONE
            return WarningState(level=NONE, previous_level=NONE, reason="Warnings disabled")

        threat = select_primary_threat(candidates)
        if threat is None:
            state = WarningState(level=NONE, previous_level=self.previous_level, reason="No active warning-level risk")
            self.previous_level = NONE
            return state

        level = RISK_TO_WARNING.get(threat.risk_level, NONE)
        previous = self.previous_level
        self.last_seen_by_track[threat.track_id] = now
        self.last_warning_by_track[threat.track_id] = level

        audio_triggered = False
        cooldown_remaining = self._cooldown_remaining(level, now)
        escalated = WARNING_ORDER[level] > WARNING_ORDER.get(previous, 0)
        if level in {CAUTION, DANGER} and (escalated or cooldown_remaining <= 0):
            audio_triggered = self.audio_player.play(level)
            self.last_audio_time_by_level[level] = now
            cooldown_remaining = self._cooldown_seconds(level)

        message = build_warning_message(threat.class_name, threat.zone, threat.risk_level)
        state = WarningState(
            level=level,
            track_id=threat.track_id,
            class_name=threat.class_name,
            message=message,
            zone=threat.zone,
            distance_m=threat.distance_m,
            ttc_seconds=threat.ttc_seconds,
            audio_triggered=audio_triggered,
            cooldown_remaining_seconds=max(0.0, cooldown_remaining),
            previous_level=previous,
            risk_level=threat.risk_level,
            reason=threat.reason,
        )
        self.previous_level = level
        return state

    def _cooldown_seconds(self, level: str) -> float:
        if level == DANGER:
            return float(self.config["high_cooldown_seconds"])
        if level == CAUTION:
            return float(self.config["medium_cooldown_seconds"])
        return 0.0

    def _cooldown_remaining(self, level: str, now: float) -> float:
        cooldown = self._cooldown_seconds(level)
        last_time = self.last_audio_time_by_level.get(level, 0.0)
        return cooldown - (now - last_time)

    def _cleanup_stale(self, now: float) -> None:
        stale_seconds = float(self.config["stale_track_seconds"])
        stale_ids = [track_id for track_id, seen in self.last_seen_by_track.items() if now - seen > stale_seconds]
        for track_id in stale_ids:
            self.last_seen_by_track.pop(track_id, None)
            self.last_warning_by_track.pop(track_id, None)


def select_primary_threat(candidates: list[WarningCandidate]) -> WarningCandidate | None:
    warning_candidates = [candidate for candidate in candidates if RISK_TO_WARNING.get(candidate.risk_level, NONE) != NONE]
    if not warning_candidates:
        return None

    def sort_key(candidate: WarningCandidate) -> tuple[int, float, float, int]:
        ttc = candidate.ttc_seconds if candidate.ttc_seconds is not None else float("inf")
        distance = candidate.distance_m if candidate.distance_m is not None else float("inf")
        return (-RISK_ORDER.get(candidate.risk_level, 0), ttc, distance, candidate.track_id)

    return sorted(warning_candidates, key=sort_key)[0]


def build_warning_message(class_name: str, zone: str, risk_level: str) -> str:
    name = class_name.upper()
    if risk_level == HIGH_RISK:
        if zone == "IN_PATH":
            return f"{name} IN VEHICLE PATH"
        return f"{name} AHEAD"
    if risk_level == MEDIUM_RISK:
        if zone == "NEAR_PATH":
            return f"{name} NEAR VEHICLE PATH"
        return f"{name} AHEAD"
    return ""
