"""Shared Airacare project configuration."""

TARGET_CLASSES = [
    "person",
    "dog",
    "cat",
    "horse",
    "cow",
    "deer",
    "goat",
]

TARGET_CLASS_SET = set(TARGET_CLASSES)
# Approximate representative object heights used by Phase 09 monocular
# distance estimation. Real people and animals vary in size, so distances
# calculated from these values are estimates, not exact measurements.
KNOWN_OBJECT_HEIGHTS_METERS = {
    "person": 1.70,
    "dog": 0.60,
    "cat": 0.30,
    "horse": 1.60,
    "cow": 1.50,
    "deer": 1.20,
    "goat": 0.75,
}

# Phase 12 prototype risk thresholds. These are transparent, configurable
# research/demo values, not official automotive safety standards.
RISK_CONFIG = {
    "high_ttc_seconds": 2.5,
    "medium_ttc_seconds": 5.0,
    "high_distance_m": 10.0,
    "medium_distance_m": 25.0,
    "fast_closing_mps": 5.0,
    "closing_mps": 0.5,
    "minimum_confidence": 0.30,
    "significant_vehicle_speed_kmh": 10.0,
    "high_vehicle_speed_kmh": 50.0,
    "risk_history_size": 5,
}

# Phase 13 warning settings. Audio is optional and non-blocking.
WARNING_CONFIG = {
    "warnings_enabled": True,
    "audio_enabled": True,
    "medium_cooldown_seconds": 3.0,
    "high_cooldown_seconds": 1.0,
    "stale_track_seconds": 2.0,
}
