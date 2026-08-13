"""Lightweight track-history state for Phase 08 movement direction."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from src.movement import MOVEMENT_THRESHOLD_PIXELS, calculate_direction


TRACK_HISTORY_LIMIT = 8
STALE_TRACK_SECONDS = 2.0


@dataclass
class TrackState:
    class_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    center_x: float
    center_y: float
    previous_center_x: float | None = None
    previous_center_y: float | None = None
    last_seen_time: float = field(default_factory=time.perf_counter)
    history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=TRACK_HISTORY_LIMIT))
    direction: str = "STATIONARY"


class TrackHistory:
    """Keep a short center-point history for each active track ID."""

    def __init__(
        self,
        history_limit: int = TRACK_HISTORY_LIMIT,
        stale_seconds: float = STALE_TRACK_SECONDS,
        movement_threshold_pixels: float = MOVEMENT_THRESHOLD_PIXELS,
    ) -> None:
        self.history_limit = history_limit
        self.stale_seconds = stale_seconds
        self.movement_threshold_pixels = movement_threshold_pixels
        self.tracks: dict[int, TrackState] = {}

    def update(
        self,
        track_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        bounding_box: tuple[float, float, float, float],
        center: tuple[float, float],
    ) -> TrackState:
        now = time.perf_counter()
        existing = self.tracks.get(track_id)

        if existing is None:
            state = TrackState(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bounding_box=bounding_box,
                center_x=center[0],
                center_y=center[1],
                last_seen_time=now,
            )
            state.history.append(center)
            self.tracks[track_id] = state
            return state

        previous_center = self._smoothed_previous_center(existing)
        existing.previous_center_x = existing.center_x
        existing.previous_center_y = existing.center_y
        existing.class_id = class_id
        existing.class_name = class_name
        existing.confidence = confidence
        existing.bounding_box = bounding_box
        existing.center_x = center[0]
        existing.center_y = center[1]
        existing.last_seen_time = now
        existing.history.append(center)
        existing.direction = calculate_direction(previous_center, center, self.movement_threshold_pixels)
        return existing

    def cleanup_stale(self) -> None:
        now = time.perf_counter()
        stale_ids = [
            track_id
            for track_id, state in self.tracks.items()
            if now - state.last_seen_time > self.stale_seconds
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

    def active_count(self) -> int:
        return len(self.tracks)

    def trail_points(self, track_id: int) -> list[tuple[int, int]]:
        state = self.tracks.get(track_id)
        if state is None:
            return []
        return [(int(x), int(y)) for x, y in state.history]

    def _smoothed_previous_center(self, state: TrackState) -> tuple[float, float] | None:
        if len(state.history) < 2:
            return (state.center_x, state.center_y)

        points = list(state.history)[:-1]
        recent = points[-min(3, len(points)) :]
        average_x = sum(point[0] for point in recent) / len(recent)
        average_y = sum(point[1] for point in recent) / len(recent)
        return average_x, average_y
