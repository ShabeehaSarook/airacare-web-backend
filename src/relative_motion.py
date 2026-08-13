"""Relative closing-speed and experimental TTC analysis for Phase 11."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


RELATIVE_HISTORY_LIMIT = 7
MIN_MOTION_SAMPLES = 3
CLOSING_THRESHOLD_MPS = 0.5
MAX_REASONABLE_DISTANCE_RATE_MPS = 35.0
STALE_MOTION_SECONDS = 2.0


@dataclass
class MotionEstimate:
    state: str = "UNKNOWN"
    closing_speed_mps: float | None = None
    ttc_seconds: float | None = None
    previous_distance_m: float | None = None
    current_distance_m: float | None = None
    dt_seconds: float | None = None
    raw_closing_speed_mps: float | None = None
    sample_count: int = 0
    outlier_rejected: bool = False


class RelativeMotionAnalyzer:
    """Store short per-track distance/time history and estimate closing trend."""

    def __init__(
        self,
        history_limit: int = RELATIVE_HISTORY_LIMIT,
        min_samples: int = MIN_MOTION_SAMPLES,
        closing_threshold_mps: float = CLOSING_THRESHOLD_MPS,
        max_distance_rate_mps: float = MAX_REASONABLE_DISTANCE_RATE_MPS,
        stale_seconds: float = STALE_MOTION_SECONDS,
    ) -> None:
        self.history_limit = history_limit
        self.min_samples = min_samples
        self.closing_threshold_mps = closing_threshold_mps
        self.max_distance_rate_mps = max_distance_rate_mps
        self.stale_seconds = stale_seconds
        self.history: dict[int, deque[tuple[float, float]]] = {}
        self.last_seen: dict[int, float] = {}

    def update(self, track_id: int, distance_m: float | None, timestamp: float | None = None) -> MotionEstimate:
        now = time.perf_counter() if timestamp is None else timestamp
        if distance_m is None or distance_m <= 0:
            return MotionEstimate()

        samples = self.history.setdefault(track_id, deque(maxlen=self.history_limit))
        previous_distance = samples[-1][1] if samples else None
        previous_time = samples[-1][0] if samples else None
        raw_closing = None
        dt = None

        if previous_time is not None and previous_distance is not None:
            dt = now - previous_time
            if dt <= 0:
                return MotionEstimate(previous_distance_m=previous_distance, current_distance_m=distance_m, sample_count=len(samples))
            raw_closing = (previous_distance - distance_m) / dt
            if abs(raw_closing) > self.max_distance_rate_mps:
                return MotionEstimate(
                    state="UNKNOWN",
                    previous_distance_m=previous_distance,
                    current_distance_m=distance_m,
                    dt_seconds=dt,
                    raw_closing_speed_mps=raw_closing,
                    sample_count=len(samples),
                    outlier_rejected=True,
                )

        samples.append((now, distance_m))
        self.last_seen[track_id] = now

        if len(samples) < self.min_samples:
            return MotionEstimate(
                state="UNKNOWN",
                previous_distance_m=previous_distance,
                current_distance_m=distance_m,
                dt_seconds=dt,
                raw_closing_speed_mps=raw_closing,
                sample_count=len(samples),
            )

        closing_speed = self._closing_speed_from_trend(samples)
        state = self._classify(closing_speed)
        ttc = distance_m / closing_speed if closing_speed > self.closing_threshold_mps else None

        return MotionEstimate(
            state=state,
            closing_speed_mps=closing_speed,
            ttc_seconds=ttc,
            previous_distance_m=previous_distance,
            current_distance_m=distance_m,
            dt_seconds=dt,
            raw_closing_speed_mps=raw_closing,
            sample_count=len(samples),
        )

    def cleanup_stale(self) -> None:
        now = time.perf_counter()
        stale_ids = [track_id for track_id, last_seen in self.last_seen.items() if now - last_seen > self.stale_seconds]
        for track_id in stale_ids:
            self.last_seen.pop(track_id, None)
            self.history.pop(track_id, None)

    def _closing_speed_from_trend(self, samples: deque[tuple[float, float]]) -> float:
        times = [sample[0] - samples[0][0] for sample in samples]
        distances = [sample[1] for sample in samples]
        mean_t = sum(times) / len(times)
        mean_d = sum(distances) / len(distances)
        denominator = sum((t - mean_t) ** 2 for t in times)
        if denominator <= 0:
            return 0.0
        slope_distance_per_second = sum((t - mean_t) * (d - mean_d) for t, d in zip(times, distances)) / denominator
        return -slope_distance_per_second

    def _classify(self, closing_speed_mps: float) -> str:
        if closing_speed_mps > self.closing_threshold_mps:
            return "CLOSING"
        if closing_speed_mps < -self.closing_threshold_mps:
            return "OPENING"
        return "STABLE"
