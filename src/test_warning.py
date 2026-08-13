"""Simulate Phase 13 driver warnings without running YOLO."""

from __future__ import annotations

import argparse
import time

from src.warning import WarningCandidate, WarningManager


RISK_BY_NAME = {
    "unknown": "UNKNOWN",
    "low": "LOW_RISK",
    "medium": "MEDIUM_RISK",
    "high": "HIGH_RISK",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Airacare warning behavior without camera or YOLO.")
    parser.add_argument("--risk", choices=sorted(RISK_BY_NAME), default="high", help="Simulated risk level.")
    parser.add_argument("--class", dest="class_name", default="dog", help="Simulated object class name.")
    parser.add_argument("--track-id", type=int, default=3, help="Simulated track ID.")
    parser.add_argument("--zone", default="IN_PATH", choices=["IN_PATH", "NEAR_PATH", "OUTSIDE_PATH"], help="Simulated path zone.")
    parser.add_argument("--distance", type=float, default=10.8, help="Simulated distance in metres.")
    parser.add_argument("--ttc", type=float, default=2.1, help="Simulated estimated TTC in seconds.")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio tone during simulation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = WarningManager(warnings_enabled=True, audio_enabled=not args.no_audio)
    candidate = WarningCandidate(
        track_id=args.track_id,
        class_name=args.class_name.lower(),
        risk_level=RISK_BY_NAME[args.risk],
        zone=args.zone,
        distance_m=args.distance,
        ttc_seconds=args.ttc,
        reason="Simulated warning test",
    )
    state = manager.update([candidate], time.perf_counter())

    print("Airacare Phase 13 Warning Test")
    print(f"Risk: {candidate.risk_level}")
    print(f"Warning level: {state.level}")
    print(f"Threat ID: {state.track_id}")
    print(f"Class: {state.class_name}")
    print(f"Message: {state.message or 'No warning'}")
    print(f"Distance: {state.distance_m if state.distance_m is not None else 'N/A'}")
    print(f"TTC: {state.ttc_seconds if state.ttc_seconds is not None else 'N/A'}")
    print(f"Audio triggered: {state.audio_triggered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
