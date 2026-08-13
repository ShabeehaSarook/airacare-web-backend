"""Airacare Animal Detector application entry point."""

from __future__ import annotations

import argparse
import sys

from src.config import WARNING_CONFIG
from src.detector import (
    CAMERA_SOURCE,
    CONFIDENCE_THRESHOLD,
    DEFAULT_TRACKER,
    DISTANCE_ENABLED_DEFAULT,
    INFERENCE_IMAGE_SIZE,
    IOU_THRESHOLD,
    run_detection,
)
from src.distance import DEFAULT_CALIBRATION_PATH
from src.fusion import DEFAULT_SENSOR_CALIBRATION_PATH
from src.path_analysis import DEFAULT_PATH_CONFIG_PATH
from src.visibility import DEFAULT_VISIBILITY_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Airacare live detection, warnings, and low-visibility optimization.")
    parser.add_argument("--camera-source", default=CAMERA_SOURCE, help="Camera index or video stream URL.")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD, help="Minimum detection confidence.")
    parser.add_argument("--imgsz", type=int, default=INFERENCE_IMAGE_SIZE, help="YOLO inference image size.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="YOLO non-max suppression IoU threshold.")
    parser.add_argument("--device", default="auto", help="YOLO device: auto, cpu, 0/cuda, or mps.")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="Ultralytics tracker config.")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH), help="Camera calibration JSON path.")
    parser.add_argument("--path-config", default=str(DEFAULT_PATH_CONFIG_PATH), help="Path-zone config JSON path.")
    parser.add_argument("--vehicle-speed-kmh", type=float, default=0.0, help="Manual test vehicle speed in km/h.")
    parser.add_argument("--enhancement", choices=["off", "auto", "lowlight"], default="auto", help="Frame enhancement mode for YOLO input.")
    parser.add_argument("--visibility-config", default=str(DEFAULT_VISIBILITY_CONFIG_PATH), help="Visibility/enhancement config JSON path.")
    parser.add_argument("--show-enhancement", action="store_true", help="Show Original | Enhanced comparison view.")
    parser.add_argument("--sensor", choices=["none", "mock", "radar", "depth", "thermal"], default="none", help="Optional Phase 15 sensor mode.")
    parser.add_argument("--sensor-data", default=None, help="Optional mock/replay sensor CSV path.")
    parser.add_argument("--sensor-port", default=None, help="Future serial sensor port, for example COM5.")
    parser.add_argument("--sensor-baud", type=int, default=115200, help="Future serial sensor baud rate.")
    parser.add_argument("--thermal-source", default=None, help="Future thermal camera source.")
    parser.add_argument("--sensor-calibration", default=str(DEFAULT_SENSOR_CALIBRATION_PATH), help="Sensor fusion calibration/config JSON path.")

    tracking_group = parser.add_mutually_exclusive_group()
    tracking_group.add_argument("--tracking", dest="tracking", action="store_true", default=True, help="Enable tracking.")
    tracking_group.add_argument("--no-tracking", dest="tracking", action="store_false", help="Disable tracking.")

    distance_group = parser.add_mutually_exclusive_group()
    distance_group.add_argument("--distance", dest="distance", action="store_true", default=DISTANCE_ENABLED_DEFAULT, help="Enable estimated distance.")
    distance_group.add_argument("--no-distance", dest="distance", action="store_false", help="Disable estimated distance.")

    warning_group = parser.add_mutually_exclusive_group()
    warning_group.add_argument("--warnings", dest="warnings", action="store_true", default=bool(WARNING_CONFIG["warnings_enabled"]), help="Enable visual driver warnings.")
    warning_group.add_argument("--no-warnings", dest="warnings", action="store_false", help="Disable visual driver warnings.")

    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--audio-warning", dest="audio_warning", action="store_true", default=bool(WARNING_CONFIG["audio_enabled"]), help="Enable non-blocking audio warning tones.")
    audio_group.add_argument("--no-audio-warning", dest="audio_warning", action="store_false", help="Disable warning audio tones.")

    parser.add_argument("--debug-detections", action="store_true", help="Print detection details.")
    parser.add_argument("--debug-distance", action="store_true", help="Print distance calculation details.")
    parser.add_argument("--debug-motion", action="store_true", help="Print closing speed and TTC details.")
    parser.add_argument("--debug-risk", action="store_true", help="Print raw and smoothed risk details.")
    parser.add_argument("--debug-warning", action="store_true", help="Print warning priority and cooldown details.")
    parser.add_argument("--debug-visibility", action="store_true", help="Print visibility and enhancement details.")
    parser.add_argument("--debug-sensors", action="store_true", help="Print Phase 15 sensor association and fusion details.")
    parser.add_argument("--no-detection", action="store_true", help="Show video only without YOLO.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(
        run_detection(
            camera_source=args.camera_source,
            confidence_threshold=args.confidence,
            inference_image_size=args.imgsz,
            iou_threshold=args.iou,
            no_detection=args.no_detection,
            requested_device=args.device,
            debug_detections=args.debug_detections,
            tracking_enabled=args.tracking,
            tracker_config=args.tracker,
            distance_enabled=args.distance,
            calibration_path=args.calibration,
            debug_distance=args.debug_distance,
            path_config=args.path_config,
            vehicle_speed_kmh=args.vehicle_speed_kmh,
            debug_motion=args.debug_motion,
            debug_risk=args.debug_risk,
            warnings_enabled=args.warnings,
            audio_warning_enabled=args.audio_warning,
            debug_warning=args.debug_warning,
            enhancement_mode=args.enhancement,
            show_enhancement=args.show_enhancement,
            visibility_config_path=args.visibility_config,
            debug_visibility=args.debug_visibility,
            sensor_mode=args.sensor,
            sensor_data=args.sensor_data,
            sensor_port=args.sensor_port,
            sensor_baud=args.sensor_baud,
            thermal_source=args.thermal_source,
            sensor_calibration_path=args.sensor_calibration,
            debug_sensors=args.debug_sensors,
        )
    )


