"""Rule-based hazard risk classification for Phase 12.

This module produces prototype estimated risk levels from existing Airacare
measurements. It does not trigger warnings or braking.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass

from src.config import RISK_CONFIG
from src.path_analysis import PATH_IN, PATH_NEAR, PATH_OUTSIDE


UNKNOWN = "UNKNOWN"
LOW_RISK = "LOW_RISK"
MEDIUM_RISK = "MEDIUM_RISK"
HIGH_RISK = "HIGH_RISK"
RISK_ORDER = {
    UNKNOWN: 0,
    LOW_RISK: 1,
    MEDIUM_RISK: 2,
    HIGH_RISK: 3,
}


@dataclass
class RiskResult:
    level: str
    score: int
    reason: str
    raw_level: str | None = None


def evaluate_risk(
    zone: str,
    distance_m: float | None,
    closing_speed_mps: float | None,
    ttc_seconds: float | None,
    movement_state: str,
    detection_confidence: float,
    vehicle_speed_kmh: float,
    config: dict | None = None,
) -> RiskResult:
    """Evaluate one tracked object's prototype hazard risk."""
    cfg = RISK_CONFIG if config is None else config

    if detection_confidence < cfg["minimum_confidence"]:
        return RiskResult(UNKNOWN, 0, "Detection confidence is below risk-evaluation threshold")

    if distance_m is None:
        return RiskResult(UNKNOWN, 0, "Distance is unavailable")

    score = 0
    reasons: list[str] = []

    if zone == PATH_IN:
        score += 35
        reasons.append("object is IN_PATH")
    elif zone == PATH_NEAR:
        score += 20
        reasons.append("object is NEAR_PATH")
    elif zone == PATH_OUTSIDE:
        score += 5
        reasons.append("object is OUTSIDE_PATH")
    else:
        score += 10
        reasons.append("path zone is unknown")

    if ttc_seconds is not None:
        if ttc_seconds <= cfg["high_ttc_seconds"]:
            score += 35
            reasons.append(f"TTC is {ttc_seconds:.1f}s")
        elif ttc_seconds <= cfg["medium_ttc_seconds"]:
            score += 22
            reasons.append(f"moderate TTC {ttc_seconds:.1f}s")
        else:
            score += 8
            reasons.append(f"longer TTC {ttc_seconds:.1f}s")
    else:
        reasons.append("TTC unavailable")

    if distance_m <= cfg["high_distance_m"]:
        score += 20
        reasons.append(f"distance is {distance_m:.1f}m")
    elif distance_m <= cfg["medium_distance_m"]:
        score += 10
        reasons.append(f"moderate distance {distance_m:.1f}m")
    else:
        reasons.append(f"far distance {distance_m:.1f}m")

    if closing_speed_mps is not None:
        if closing_speed_mps >= cfg["fast_closing_mps"]:
            score += 20
            reasons.append(f"fast closing {closing_speed_mps:.1f}m/s")
        elif closing_speed_mps >= cfg["closing_mps"]:
            score += 10
            reasons.append(f"closing {closing_speed_mps:.1f}m/s")
        elif closing_speed_mps <= -cfg["closing_mps"]:
            score -= 12
            reasons.append("object is opening")
        else:
            reasons.append("relative distance is stable")
    else:
        reasons.append("closing speed unavailable")

    if vehicle_speed_kmh >= cfg["high_vehicle_speed_kmh"]:
        score += 10
        reasons.append(f"vehicle speed is {vehicle_speed_kmh:.0f}km/h")
    elif vehicle_speed_kmh >= cfg["significant_vehicle_speed_kmh"]:
        score += 5
        reasons.append(f"vehicle speed is {vehicle_speed_kmh:.0f}km/h")

    if vehicle_speed_kmh <= 0 and movement_state != "CLOSING":
        score = min(score, 35)
        reasons.append("vehicle speed is zero and object is not closing")

    if zone == PATH_OUTSIDE and movement_state == "OPENING":
        score = min(score, 25)
        reasons.append("outside path and opening")

    score = max(0, min(100, score))
    level = _level_from_score(score)

    if zone == PATH_IN and movement_state == "CLOSING" and ttc_seconds is not None:
        if ttc_seconds <= cfg["high_ttc_seconds"]:
            level = HIGH_RISK
            score = max(score, 80)
        elif ttc_seconds <= cfg["medium_ttc_seconds"]:
            level = max_level(level, MEDIUM_RISK)
            score = max(score, 55)

    if zone == PATH_NEAR and movement_state == "CLOSING" and distance_m <= cfg["high_distance_m"]:
        level = max_level(level, MEDIUM_RISK)
        score = max(score, 50)

    reason = "; ".join(reasons[:3]) if reasons else "No risk factors available"
    return RiskResult(level, score, reason)


def _level_from_score(score: int) -> str:
    if score >= 70:
        return HIGH_RISK
    if score >= 30:
        return MEDIUM_RISK
    return LOW_RISK


def max_level(left: str, right: str) -> str:
    return left if RISK_ORDER[left] >= RISK_ORDER[right] else right


def scene_risk(risks: list[RiskResult]) -> str:
    if not risks:
        return UNKNOWN
    return max((risk.level for risk in risks), key=lambda level: RISK_ORDER.get(level, 0))


class RiskSmoother:
    """Per-track risk smoothing with fast escalation and slower de-escalation."""

    def __init__(self, history_size: int | None = None, stale_seconds: float = 2.0) -> None:
        self.history_size = history_size or int(RISK_CONFIG["risk_history_size"])
        self.stale_seconds = stale_seconds
        self.history: dict[int, deque[str]] = {}
        self.last_seen: dict[int, float] = {}
        self.current: dict[int, str] = {}

    def update(self, track_id: int, raw_level: str, timestamp: float | None = None) -> str:
        now = time.perf_counter() if timestamp is None else timestamp
        previous = self.current.get(track_id, UNKNOWN)
        history = self.history.setdefault(track_id, deque(maxlen=self.history_size))
        history.append(raw_level)
        self.last_seen[track_id] = now

        if RISK_ORDER.get(raw_level, 0) > RISK_ORDER.get(previous, 0):
            smoothed = raw_level
        else:
            counts = Counter(history)
            smoothed = max(counts, key=lambda level: (counts[level], RISK_ORDER.get(level, 0)))
            if RISK_ORDER.get(previous, 0) - RISK_ORDER.get(raw_level, 0) >= 2:
                recent_safe = sum(1 for level in history if RISK_ORDER.get(level, 0) <= RISK_ORDER[LOW_RISK])
                if recent_safe < 3:
                    smoothed = previous

        self.current[track_id] = smoothed
        return smoothed

    def cleanup_stale(self) -> None:
        now = time.perf_counter()
        stale_ids = [track_id for track_id, last_seen in self.last_seen.items() if now - last_seen > self.stale_seconds]
        for track_id in stale_ids:
            self.last_seen.pop(track_id, None)
            self.history.pop(track_id, None)
            self.current.pop(track_id, None)
