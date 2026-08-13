"""Airacare Animal Detector - Phase 15 optional sensor fusion."""

from __future__ import annotations

import os
import re
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import cv2
from ultralytics import YOLO

from src.config import KNOWN_OBJECT_HEIGHTS_METERS, TARGET_CLASSES, TARGET_CLASS_SET, WARNING_CONFIG
from src.distance import (
    DEFAULT_CALIBRATION_PATH,
    DistanceHistory,
    estimate_distance,
    focal_length_for_runtime_resolution,
    load_calibration,
)
from src.fusion import CameraObjectState, SensorFusionEngine, DEFAULT_SENSOR_CALIBRATION_PATH, load_sensor_fusion_config
from src.image_enhancement import enhance_frame
from src.movement import MOVEMENT_THRESHOLD_PIXELS
from src.path_analysis import DEFAULT_PATH_CONFIG_PATH, PathAnalyzer
from src.relative_motion import CLOSING_THRESHOLD_MPS, RELATIVE_HISTORY_LIMIT, RelativeMotionAnalyzer
from src.risk import RiskResult, RiskSmoother, evaluate_risk, scene_risk
from src.sensors.base import SensorSource, SensorStatus
from src.sensors.depth import DepthSensorSource
from src.sensors.mock_sensor import MockSensorSource
from src.sensors.radar import RadarSensorSource
from src.sensors.thermal import ThermalSensorSource
from src.tracker import TrackHistory
from src.vehicle_speed import ManualSpeedProvider
from src.visibility import DEFAULT_VISIBILITY_CONFIG_PATH, VERY_DARK, analyze_visibility, load_visibility_config
from src.warning import WarningCandidate, WarningManager, WarningState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WINDOW_TITLE = "Airacare Animal Detector - Phase 15"
CUSTOM_MODEL_PATH = PROJECT_ROOT / "models" / "airacare_animal_detector_best.pt"
PRETRAINED_FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CAMERA_INDEX = 0
CAMERA_SOURCE = os.getenv("AIRACARE_CAMERA_SOURCE", str(CAMERA_INDEX))
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
INFERENCE_IMAGE_SIZE = 640
DEFAULT_TRACKER = "bytetrack.yaml"
DISTANCE_ENABLED_DEFAULT = True
FIRST_FRAME_TIMEOUT_SECONDS = 10.0
NETWORK_CAMERA_CHECK_TIMEOUT_SECONDS = 3.0
STALE_FRAME_WARNING_SECONDS = 3.0
CONSOLE_LOG_INTERVAL_SECONDS = 2.0
LIVE_CAPTURE_DIR = PROJECT_ROOT / "results" / "phase15_sensor_captures"

BOX_COLOR = (0, 220, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND = (0, 0, 0)
HEADER_COLOR = (255, 255, 255)
TRAIL_COLOR = (255, 200, 0)
RISK_COLORS = {
    "UNKNOWN": (180, 180, 180),
    "LOW_RISK": (0, 220, 0),
    "MEDIUM_RISK": (0, 200, 255),
    "HIGH_RISK": (0, 0, 255),
}


class LatestFrameCapture:
    """Read frames in the background and retain only the newest frame."""

    def __init__(self, source: str | int, is_network_source: bool) -> None:
        self.source = source
        self.is_network_source = is_network_source
        self.capture: cv2.VideoCapture | None = None
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.running = False
        self.latest_frame = None
        self.latest_frame_id = 0
        self.last_frame_time = 0.0
        self.last_error: str | None = None

    def open(self) -> bool:
        if self.is_network_source:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay|max_delay;0")

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            return False

        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.is_network_source:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return True

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, name="LatestFrameCapture", daemon=True)
        self.thread.start()

    def _reader_loop(self) -> None:
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.last_error = "Could not read frame from camera source."
                time.sleep(0.01)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.last_frame_time = time.perf_counter()
                self.last_error = None

    def wait_for_frame(self, timeout_seconds: float) -> bool:
        deadline = time.perf_counter() + timeout_seconds
        while time.perf_counter() < deadline:
            with self.lock:
                if self.latest_frame is not None:
                    return True
            time.sleep(0.03)
        return False

    def get_latest_frame(self) -> tuple[bool, Any, int, float]:
        with self.lock:
            if self.latest_frame is None:
                return False, None, self.latest_frame_id, self.last_frame_time
            return True, self.latest_frame.copy(), self.latest_frame_id, self.last_frame_time

    def is_running(self) -> bool:
        return self.running

    def release(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def normalize_model_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name).lower() for class_id, name in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(name).lower() for index, name in enumerate(names)}
    return {}


def resolve_device(requested_device: str = "auto") -> str:
    requested = requested_device.lower().strip()
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"


def sanitize_camera_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source

    source_text = str(source).strip().strip('"').strip("'")
    markdown_match = re.fullmatch(
        r"\[(?P<label>https?://[^\]]+|rtsp://[^\]]+)\]\((?P<url>https?://[^)]+|rtsp://[^)]+)\)",
        source_text,
    )
    if markdown_match:
        return markdown_match.group("url")

    if source_text.isdigit():
        return int(source_text)

    return source_text


def normalize_network_camera_url(source: str | int) -> str | int:
    """Normalize plain IP Webcam base URLs without changing local camera IDs."""
    if not is_network_camera_source(source):
        return source

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.path in {"", "/"}:
        normalized = urlunparse(parsed._replace(path="/video"))
        print(f"Phone camera URL did not include a stream path; using: {normalized}")
        return normalized
    return source


def check_network_camera_reachable(source: str | int, timeout_seconds: float = NETWORK_CAMERA_CHECK_TIMEOUT_SECONDS) -> bool:
    """Check that a plain HTTP phone-camera server is reachable before OpenCV opens it."""
    if not isinstance(source, str) or not source.lower().startswith(("http://", "https://")):
        return True

    parsed = urlparse(source)
    base_url = urlunparse((parsed.scheme, parsed.netloc, "/", "", "", ""))
    request = Request(base_url, headers={"User-Agent": "AiracareAnimalDetector/1.0"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 500
    except HTTPError as error:
        return 200 <= int(error.code) < 500
    except (OSError, URLError, TimeoutError) as error:
        print(f"Network camera reachability check failed for {base_url}")
        print(f"Details: {error}")
        return False
def is_network_camera_source(source: str | int) -> bool:
    return isinstance(source, str) and source.lower().startswith(("http://", "https://", "rtsp://"))


def source_label(is_network_source: bool) -> str:
    return "PHONE" if is_network_source else "WEBCAM"


def create_sensor_source(
    sensor_mode: str,
    sensor_data: str | None = None,
    sensor_port: str | None = None,
    sensor_baud: int = 115200,
    thermal_source: str | None = None,
) -> SensorSource | None:
    mode = sensor_mode.lower().strip()
    if mode == "none":
        return None
    if mode == "mock":
        return MockSensorSource(sensor_data)
    if mode == "radar":
        return RadarSensorSource(port=sensor_port, baud=sensor_baud)
    if mode == "depth":
        return DepthSensorSource()
    if mode == "thermal":
        return ThermalSensorSource(source=thermal_source)
    raise ValueError(f"Unsupported sensor mode: {sensor_mode}")


def sensor_mode_label(sensor_mode: str) -> str:
    mode = sensor_mode.lower().strip()
    if mode == "none":
        return "CAMERA_ONLY"
    if mode == "mock":
        return "CAMERA + MOCK_RADAR"
    return f"CAMERA + {mode.upper()}"


def format_source_tag(source: str) -> str:
    return source.replace("_", " ")

def verify_custom_model_classes(model: Any) -> bool:
    model_names = normalize_model_names(getattr(model, "names", {}))
    expected_names = {index: name for index, name in enumerate(TARGET_CLASSES)}

    print("Loaded model classes:")
    if not model_names:
        print("- none reported")
        print("Error: The loaded model does not expose class names.")
        return False

    for class_id in sorted(model_names):
        print(f"- {class_id}: {model_names[class_id]}")

    if model_names != expected_names:
        print("Error: Custom model classes do not exactly match Airacare target IDs.")
        print(f"Expected: {expected_names}")
        print(f"Loaded:   {model_names}")
        return False

    return True


def print_pretrained_target_support(model: Any) -> None:
    model_names = set(normalize_model_names(getattr(model, "names", {})).values())
    supported = [name for name in TARGET_CLASSES if name in model_names]
    unsupported = [name for name in TARGET_CLASSES if name not in model_names]
    print("Loaded pretrained model classes for Airacare demo:")
    print("Supported: " + (", ".join(supported) if supported else "none"))
    print("Unsupported until custom training: " + (", ".join(unsupported) if unsupported else "none"))
    print("Deer/goat detections are not faked.")
    print()


def load_detection_model() -> tuple[Any | None, bool, Path | None]:
    if CUSTOM_MODEL_PATH.exists():
        try:
            model = YOLO(str(CUSTOM_MODEL_PATH))
        except Exception as error:
            print(f"Error: Could not load custom YOLO model '{CUSTOM_MODEL_PATH}'.")
            print(f"Details: {error}")
            return None, False, CUSTOM_MODEL_PATH

        if not verify_custom_model_classes(model):
            return None, False, CUSTOM_MODEL_PATH
        return model, False, CUSTOM_MODEL_PATH

    print(f"Warning: Custom model not found: {CUSTOM_MODEL_PATH}")
    print(f"Using pretrained demo model instead: {PRETRAINED_FALLBACK_MODEL_PATH}")
    print("Airacare-only deer/goat detection requires the trained 7-class model.")
    if not PRETRAINED_FALLBACK_MODEL_PATH.exists():
        print(f"Error: Pretrained fallback model not found: {PRETRAINED_FALLBACK_MODEL_PATH}")
        return None, True, PRETRAINED_FALLBACK_MODEL_PATH

    try:
        model = YOLO(str(PRETRAINED_FALLBACK_MODEL_PATH))
    except Exception as error:
        print(f"Error: Could not load pretrained YOLO model '{PRETRAINED_FALLBACK_MODEL_PATH}'.")
        print(f"Details: {error}")
        return None, True, PRETRAINED_FALLBACK_MODEL_PATH

    print_pretrained_target_support(model)
    return model, True, PRETRAINED_FALLBACK_MODEL_PATH


def draw_text_with_background(frame, text: str, position: tuple[int, int], color=TEXT_COLOR) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 2
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 6),
        (x + text_width + 8, y + baseline - 2),
        TEXT_BACKGROUND,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, color, thickness, cv2.LINE_AA)


def draw_header(
    frame,
    fps: float | None,
    inference_ms: float | None,
    source_name: str,
    no_detection: bool,
    active_tracks: int,
    distance_active: bool,
    vehicle_speed_kmh: float,
    vehicle_speed_source: str,
    current_scene_risk: str,
    warnings_enabled: bool,
    audio_enabled: bool,
    visibility_class: str,
    enhancement_active: bool,
    enhancement_ms: float | None,
    sensor_mode: str,
    sensor_status: SensorStatus | None,
) -> None:
    draw_text_with_background(frame, "Airacare Animal Detector", (20, 35), HEADER_COLOR)
    draw_text_with_background(frame, "Phase 15 - Optional Sensor Fusion", (20, 70), HEADER_COLOR)
    draw_text_with_background(frame, f"Source: {source_name}", (20, 105), HEADER_COLOR)
    draw_text_with_background(frame, "Video only - no YOLO" if no_detection else "Press Q to Quit", (20, 140), HEADER_COLOR)
    if fps is not None:
        draw_text_with_background(frame, f"FPS: {fps:.1f}", (20, 175), HEADER_COLOR)
    if inference_ms is not None and not no_detection:
        draw_text_with_background(frame, f"Inference: {inference_ms:.0f} ms", (20, 210), HEADER_COLOR)
    if not no_detection:
        draw_text_with_background(frame, f"Tracks: {active_tracks}", (20, 245), HEADER_COLOR)
        draw_text_with_background(frame, f"Distance: {'ON' if distance_active else 'OFF'}", (20, 280), HEADER_COLOR)
        draw_text_with_background(frame, f"Vehicle Speed ({vehicle_speed_source}): {vehicle_speed_kmh:.0f} km/h", (20, 315), HEADER_COLOR)
        draw_text_with_background(frame, f"Scene Risk: {current_scene_risk}", (20, 350), RISK_COLORS.get(current_scene_risk, HEADER_COLOR))
        draw_text_with_background(frame, f"Warnings: {'ON' if warnings_enabled else 'OFF'}", (20, 385), HEADER_COLOR)
        draw_text_with_background(frame, f"Audio: {'ON' if audio_enabled else 'OFF'}", (20, 420), HEADER_COLOR)
        draw_text_with_background(frame, f"Visibility: {visibility_class}", (20, 455), HEADER_COLOR)
        draw_text_with_background(frame, f"Enhancement: {'ON' if enhancement_active else 'OFF'}", (20, 490), HEADER_COLOR)
        if enhancement_ms is not None:
            draw_text_with_background(frame, f"Enhance: {enhancement_ms:.1f} ms", (20, 525), HEADER_COLOR)
        sensor_text = sensor_mode if sensor_status is None else f"{sensor_mode} ({sensor_status.state})"
        draw_text_with_background(frame, f"Sensor: {sensor_text}", (20, 560), HEADER_COLOR)
        if visibility_class == VERY_DARK:
            draw_text_with_background(frame, "CAMERA VISIBILITY POOR", (20, 595), (0, 200, 255))


def draw_detection(frame, box: Iterable[float], label_lines: list[str], color=BOX_COLOR) -> None:
    x1, y1, x2, y2 = [int(value) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label_y = max(y1 - 8 - (len(label_lines) - 1) * 26, 20)
    for line in label_lines:
        draw_text_with_background(frame, line, (x1, label_y), color=color)
        label_y += 26


def draw_trail(frame, points: list[tuple[int, int]]) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, TRAIL_COLOR, 2)


def draw_warning_banner(frame, warning_state: WarningState) -> None:
    if warning_state.level == "NONE":
        return

    height, width = frame.shape[:2]
    banner_width = min(width - 80, 720)
    x1 = max(40, (width - banner_width) // 2)
    y1 = 24
    x2 = x1 + banner_width
    y2 = 150
    color = (0, 0, 255) if warning_state.level == "DANGER" else (0, 200, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    title = "!!! DANGER !!!" if warning_state.level == "DANGER" else "CAUTION"
    lines = [title, warning_state.message]
    if warning_state.distance_m is not None:
        lines.append(f"Distance ~{warning_state.distance_m:.1f}m")
    if warning_state.ttc_seconds is not None:
        lines.append(f"TTC ~{warning_state.ttc_seconds:.1f}s")
    else:
        lines.append("TTC N/A")

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y1 + 34
    for index, line in enumerate(lines):
        scale = 0.85 if index == 0 else 0.65
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(line, font, scale, thickness)
        x = x1 + max(10, (banner_width - text_width) // 2)
        cv2.putText(frame, line, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += text_height + 10


def draw_detection_summary(frame, counts: Counter[str]) -> None:
    y = 665
    if not counts:
        draw_text_with_background(frame, "Detected: none", (20, y), HEADER_COLOR)
        return

    draw_text_with_background(frame, "Detected:", (20, y), HEADER_COLOR)
    y += 35
    for class_name in TARGET_CLASSES:
        count = counts.get(class_name, 0)
        if count:
            draw_text_with_background(frame, f"{class_name.title()}: {count}", (20, y), HEADER_COLOR)
            y += 35


def save_detected_frame(frame, original_frame=None, enhanced_frame=None) -> Path:
    LIVE_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = LIVE_CAPTURE_DIR / f"phase15_annotated_{timestamp}.jpg"
    cv2.imwrite(str(output_path), frame)
    if original_frame is not None:
        cv2.imwrite(str(LIVE_CAPTURE_DIR / f"phase15_original_{timestamp}.jpg"), original_frame)
    if enhanced_frame is not None:
        cv2.imwrite(str(LIVE_CAPTURE_DIR / f"phase15_enhanced_{timestamp}.jpg"), enhanced_frame)
    return output_path


def print_connection_help(is_network_source: bool) -> None:
    if is_network_source:
        print("Unable to receive frames from phone camera.")
        print("Check:")
        print("- phone and laptop are on same Wi-Fi")
        print("- IP Webcam server is running")
        print("- laptop browser can open the phone server page")
        print("- use the plain stream URL, for example: http://192.168.1.230:8080/video")
        print("- do not paste Markdown links like [url](url)")
    else:
        print("Unable to receive frames from webcam. Check that the webcam is connected and not used by another app.")


def extract_track_id(detected_box: Any, fallback_id: int) -> int:
    box_id = getattr(detected_box, "id", None)
    if box_id is None:
        return fallback_id
    try:
        return int(box_id[0].item())
    except (TypeError, ValueError, IndexError, AttributeError):
        return fallback_id


def format_debug_value(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def format_ttc(value: float | None) -> str:
    return "TTC:N/A" if value is None else f"TTC:~{value:.1f}s"


def format_closing(value: float | None) -> str:
    return "EST.CLOSE:N/A" if value is None else f"EST.CLOSE:{value:.1f}m/s"


def format_risk_label(level: str) -> str:
    if level.endswith("_RISK"):
        return level.replace("_RISK", "")
    return level


def run_detection(
    camera_source: str | int | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    inference_image_size: int = INFERENCE_IMAGE_SIZE,
    iou_threshold: float = IOU_THRESHOLD,
    no_detection: bool = False,
    requested_device: str = "auto",
    debug_detections: bool = False,
    tracking_enabled: bool = True,
    tracker_config: str = DEFAULT_TRACKER,
    distance_enabled: bool = DISTANCE_ENABLED_DEFAULT,
    calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
    debug_distance: bool = False,
    path_config: str | Path = DEFAULT_PATH_CONFIG_PATH,
    vehicle_speed_kmh: float = 0.0,
    debug_motion: bool = False,
    debug_risk: bool = False,
    warnings_enabled: bool = True,
    audio_warning_enabled: bool = True,
    debug_warning: bool = False,
    enhancement_mode: str = "auto",
    show_enhancement: bool = False,
    visibility_config_path: str | Path = DEFAULT_VISIBILITY_CONFIG_PATH,
    debug_visibility: bool = False,
    sensor_mode: str = "none",
    sensor_data: str | None = None,
    sensor_port: str | None = None,
    sensor_baud: int = 115200,
    thermal_source: str | None = None,
    sensor_calibration_path: str | Path = DEFAULT_SENSOR_CALIBRATION_PATH,
    debug_sensors: bool = False,
) -> int:
    selected_camera_source = sanitize_camera_source(camera_source if camera_source is not None else CAMERA_SOURCE)
    selected_camera_source = normalize_network_camera_url(selected_camera_source)
    is_network_source = is_network_camera_source(selected_camera_source)
    source_name = source_label(is_network_source)
    device = resolve_device(requested_device)
    speed_provider = ManualSpeedProvider(vehicle_speed_kmh)
    vehicle_speed_mps = speed_provider.get_vehicle_speed_mps()

    print("Airacare Animal Detector")
    print("Phase 15: Optional Sensor Integration + Sensor Fusion")
    print(f"Camera source: {source_name}")
    print(f"Camera source value: {selected_camera_source}")
    print(f"Model: {CUSTOM_MODEL_PATH if CUSTOM_MODEL_PATH.exists() else PRETRAINED_FALLBACK_MODEL_PATH}")
    print(f"Device: {device.upper() if device != '0' else 'CUDA'}")
    print(f"Inference size: {inference_image_size}")
    print(f"Confidence threshold: {confidence_threshold:.2f}")
    print(f"IoU threshold: {iou_threshold:.2f}")
    print(f"Tracking: {tracking_enabled}")
    print(f"Tracker: {tracker_config if tracking_enabled else 'disabled'}")
    print(f"Movement threshold: {MOVEMENT_THRESHOLD_PIXELS} px")
    print(f"Path config: {path_config}")
    print(f"Vehicle Speed ({speed_provider.source_name}): {speed_provider.get_vehicle_speed_kmh():.1f} km/h ({vehicle_speed_mps:.2f} m/s)")
    print("Vehicle speed is metadata only; relative closing speed comes from measured distance changes.")
    print("Distance, closing speed, and TTC are estimates based on monocular camera calibration.")
    print("Phase 14 improves RGB frame visibility when possible, but does not create thermal/night-vision capability.")
    print(f"Distance requested: {distance_enabled}")
    print(f"Calibration file: {calibration_path}")
    print(f"Closing threshold: {CLOSING_THRESHOLD_MPS:.2f} m/s")
    print(f"Relative motion history: {RELATIVE_HISTORY_LIMIT} samples")
    print(f"Debug detections: {debug_detections}")
    print(f"Debug distance: {debug_distance}")
    print(f"Debug motion: {debug_motion}")
    print(f"Debug risk: {debug_risk}")
    print(f"Warnings enabled: {warnings_enabled}")
    print(f"Audio warning enabled: {audio_warning_enabled}")
    print(f"Debug warning: {debug_warning}")
    print(f"Enhancement mode: {enhancement_mode}")
    print(f"Show enhancement comparison: {show_enhancement}")
    print(f"Visibility config: {visibility_config_path}")
    print(f"Debug visibility: {debug_visibility}")
    print(f"Sensor mode: {sensor_mode_label(sensor_mode)}")
    print(f"Sensor calibration: {sensor_calibration_path}")
    print(f"Sensor data: {sensor_data if sensor_data else 'none'}")
    print(f"Debug sensors: {debug_sensors}")
    print("External sensors are optional; camera-only fallback remains active.")
    print()

    if is_network_source and not check_network_camera_reachable(selected_camera_source):
        print_connection_help(True)
        return 1
    model = None
    using_pretrained_fallback = False
    if not no_detection:
        print("Loading YOLO model...")
        model, using_pretrained_fallback, loaded_model_path = load_detection_model()
        if model is None:
            return 1
        print(f"YOLO model loaded successfully: {loaded_model_path}")
    else:
        print("No-detection mode enabled. Opening video only; YOLO will not run.")

    try:
        path_analyzer = PathAnalyzer.from_file(path_config)
    except Exception as error:
        print(f"Path config could not be loaded; using default path zones. Details: {error}")
        path_analyzer = PathAnalyzer()

    calibration = None
    distance_active = False
    runtime_focal_length = None
    if distance_enabled and not no_detection:
        try:
            calibration = load_calibration(calibration_path)
        except Exception as error:
            print(f"Distance estimation disabled: invalid calibration file: {error}")
        if calibration is None:
            print("Distance estimation disabled: camera calibration file not found.")
        else:
            distance_active = True
            print(f"Distance calibration loaded: focal_length={float(calibration['focal_length_pixels']):.2f} px")
            print(f"Known class height assumptions: {KNOWN_OBJECT_HEIGHTS_METERS}")

    try:
        visibility_config = load_visibility_config(visibility_config_path)
    except Exception as error:
        print(f"Visibility config could not be loaded; using defaults. Details: {error}")
        visibility_config = load_visibility_config("__missing_visibility_config__.json")

    print("Connecting to phone camera..." if is_network_source else "Opening webcam...")
    capture = LatestFrameCapture(selected_camera_source, is_network_source)
    track_history = TrackHistory(movement_threshold_pixels=MOVEMENT_THRESHOLD_PIXELS)
    distance_history = DistanceHistory()
    motion_analyzer = RelativeMotionAnalyzer()
    risk_smoother = RiskSmoother()
    warning_manager = WarningManager(warnings_enabled=warnings_enabled, audio_enabled=audio_warning_enabled)
    sensor_source = create_sensor_source(sensor_mode, sensor_data, sensor_port, sensor_baud, thermal_source)
    sensor_status: SensorStatus | None = None
    sensor_mode_text = sensor_mode_label(sensor_mode)
    sensor_fusion_config = load_sensor_fusion_config(sensor_calibration_path)
    fusion_engine = SensorFusionEngine(sensor_fusion_config)
    if sensor_source is not None:
        sensor_source.start()
        sensor_status = sensor_source.status()
        print(f"Sensor status: {sensor_status.sensor_type} {sensor_status.state} - {sensor_status.message}")
    else:
        print("Sensor status: CAMERA_ONLY")

    try:
        if not capture.open():
            print(f"Error: Could not open camera source: {selected_camera_source}")
            print_connection_help(is_network_source)
            return 1

        print("Waiting for first phone-camera frame..." if is_network_source else "Waiting for first webcam frame...")
        capture.start()
        if not capture.wait_for_frame(FIRST_FRAME_TIMEOUT_SECONDS):
            print_connection_help(is_network_source)
            return 1

        print("Phone camera connected." if is_network_source else "Webcam connected.")
        print("Press Q to quit. Press S to save the current warning frame.")

        previous_display_time = time.perf_counter()
        last_console_log_time = time.perf_counter()
        last_frame_warning_time = 0.0
        last_processed_frame_id = -1
        fps = None
        inference_ms = None

        while capture.is_running():
            ok, frame, frame_id, frame_time = capture.get_latest_frame()
            if not ok:
                time.sleep(0.01)
                continue

            if frame_id == last_processed_frame_id:
                time.sleep(0.005)
                continue
            last_processed_frame_id = frame_id

            now = time.perf_counter()
            if frame_time and now - frame_time > STALE_FRAME_WARNING_SECONDS and now - last_frame_warning_time > STALE_FRAME_WARNING_SECONDS:
                print("Warning: camera stream appears delayed or stalled.")
                last_frame_warning_time = now

            original_frame = frame.copy()
            visibility_metrics = analyze_visibility(frame, visibility_config)
            enhancement_result = enhance_frame(frame, visibility_metrics.visibility_class, enhancement_mode, visibility_config)
            yolo_frame = enhancement_result.frame
            enhancement_ms = enhancement_result.elapsed_ms
            if debug_visibility:
                print(
                    f"Visibility: {visibility_metrics.visibility_class} "
                    f"Brightness: {visibility_metrics.brightness:.1f} "
                    f"Contrast: {visibility_metrics.contrast:.1f} "
                    f"Blur: {visibility_metrics.blur_score:.1f} "
                    f"Gamma: {enhancement_result.gamma if enhancement_result.gamma is not None else 'N/A'} "
                    f"CLAHE: {'ON' if enhancement_result.clahe_enabled else 'OFF'} "
                    f"Enhancement: {enhancement_ms:.1f} ms"
                )
            if enhancement_ms > float(visibility_config.get("max_enhancement_time_warning_ms", 25.0)):
                print(f"Warning: enhancement preprocessing is slow ({enhancement_ms:.1f} ms).")

            frame_height, frame_width = frame.shape[:2]
            sensor_measurements = sensor_source.get_latest_measurements(now) if sensor_source is not None else []
            sensor_status = sensor_source.status(now) if sensor_source is not None else None
            if distance_active and calibration is not None and runtime_focal_length is None:
                runtime_focal_length, warning = focal_length_for_runtime_resolution(calibration, (frame_width, frame_height))
                if warning:
                    print(f"Distance warning: {warning}")
                print(f"Runtime focal length: {runtime_focal_length:.2f} px for {frame_width}x{frame_height}")

            counts: Counter[str] = Counter()
            frame_risks: list[RiskResult] = []
            warning_candidates: list[WarningCandidate] = []

            if not no_detection and model is not None:
                try:
                    inference_start = time.perf_counter()
                    if tracking_enabled:
                        results = model.track(
                            source=yolo_frame,
                            conf=confidence_threshold,
                            imgsz=inference_image_size,
                            iou=iou_threshold,
                            device=device,
                            tracker=tracker_config,
                            persist=True,
                            verbose=False,
                        )
                    else:
                        results = model.predict(
                            source=yolo_frame,
                            conf=confidence_threshold,
                            imgsz=inference_image_size,
                            iou=iou_threshold,
                            device=device,
                            verbose=False,
                        )
                    inference_ms = (time.perf_counter() - inference_start) * 1000
                except Exception as error:
                    print(f"Inference/tracking error: {error}")
                    break

                result = results[0] if results else None
                boxes = getattr(result, "boxes", []) if result is not None else []
                model_names = normalize_model_names(getattr(model, "names", {}))

                if debug_detections:
                    print("Detected:")

                used_sensor_keys: set[str | int] = set()
                for index, detected_box in enumerate(boxes, start=1):
                    confidence = float(detected_box.conf[0])
                    class_id = int(detected_box.cls[0])
                    class_name = model_names.get(class_id, f"class_{class_id}").lower()
                    coordinates = tuple(float(value) for value in detected_box.xyxy[0].tolist())

                    if class_name not in TARGET_CLASS_SET:
                        continue
                    if using_pretrained_fallback and class_name in {"deer", "goat"}:
                        continue

                    x1, y1, x2, y2 = coordinates
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    track_id = extract_track_id(detected_box, fallback_id=-index)
                    state = track_history.update(track_id, class_id, class_name, confidence, coordinates, center)
                    path_zone = path_analyzer.classify(center, frame_width, frame_height)
                    counts[class_name] += 1

                    bbox_height = max(0.0, y2 - y1)
                    raw_distance = (
                        estimate_distance(class_name, bbox_height, runtime_focal_length)
                        if distance_active and runtime_focal_length is not None
                        else None
                    )
                    smoothed_distance = distance_history.update(track_id, raw_distance)
                    motion = motion_analyzer.update(track_id, smoothed_distance, now)
                    camera_state = CameraObjectState(
                        track_id=track_id,
                        class_name=class_name,
                        detection_confidence=confidence,
                        bbox=coordinates,
                        center=center,
                        distance_camera_m=smoothed_distance,
                        camera_closing_speed_mps=motion.closing_speed_mps,
                        movement=motion.state,
                        path_zone=path_zone,
                        timestamp=now,
                    )
                    available_sensor_measurements = [
                        measurement for measurement in sensor_measurements
                        if (measurement.sensor_object_id if measurement.sensor_object_id is not None else id(measurement)) not in used_sensor_keys
                    ]
                    fused_state = fusion_engine.fuse(camera_state, available_sensor_measurements, frame_width)
                    if fused_state.association_status == "ASSOCIATED":
                        used_sensor_keys.add(fused_state.sensor_object_id if fused_state.sensor_object_id is not None else id(fused_state))
                    fused_distance = fused_state.fused_distance_m
                    fused_closing = fused_state.fused_closing_speed_mps
                    fused_ttc = fused_state.ttc_seconds
                    risk_motion_state = motion.state
                    if fused_closing is not None:
                        if fused_closing > CLOSING_THRESHOLD_MPS:
                            risk_motion_state = "CLOSING"
                        elif fused_closing < -CLOSING_THRESHOLD_MPS:
                            risk_motion_state = "OPENING"
                        else:
                            risk_motion_state = "STABLE"

                    distance_label = (
                        f"DIST:~{fused_distance:.1f}m [{format_source_tag(fused_state.distance_source)}]"
                        if fused_distance is not None
                        else "DIST:N/A"
                    )
                    motion_label = f"{format_closing(fused_closing)} [{risk_motion_state}] [{format_source_tag(fused_state.closing_speed_source)}]"
                    ttc_label = format_ttc(fused_ttc)
                    raw_risk = evaluate_risk(
                        zone=path_zone,
                        distance_m=fused_distance,
                        closing_speed_mps=fused_closing,
                        ttc_seconds=fused_ttc,
                        movement_state=risk_motion_state,
                        detection_confidence=confidence,
                        vehicle_speed_kmh=speed_provider.get_vehicle_speed_kmh(),
                    )
                    smoothed_level = risk_smoother.update(track_id, raw_risk.level, now)
                    risk_result = RiskResult(smoothed_level, raw_risk.score, raw_risk.reason, raw_level=raw_risk.level)
                    frame_risks.append(risk_result)
                    warning_candidates.append(
                        WarningCandidate(
                            track_id=track_id,
                            class_name=class_name,
                            risk_level=smoothed_level,
                            zone=path_zone,
                            distance_m=fused_distance,
                            ttc_seconds=fused_ttc,
                            reason=raw_risk.reason,
                        )
                    )

                    label_lines = [
                        f"{class_name.upper()} ID:{track_id} {confidence * 100:.0f}% {path_zone}",
                        f"{distance_label} {motion_label}",
                        f"{ttc_label} RISK:{format_risk_label(smoothed_level)}",
                    ]
                    draw_detection(frame, coordinates, label_lines, color=RISK_COLORS.get(smoothed_level, BOX_COLOR))
                    draw_trail(frame, track_history.trail_points(track_id))

                    if debug_detections:
                        rounded_box = tuple(round(value, 1) for value in coordinates)
                        print(
                            f"{class_name} id={track_id} confidence={confidence:.2f} "
                            f"direction={state.direction} zone={path_zone} box={rounded_box}"
                        )
                    if debug_distance and distance_enabled:
                        known_height = KNOWN_OBJECT_HEIGHTS_METERS.get(class_name)
                        print(
                            f"Distance debug: track_id={track_id} class={class_name} "
                            f"bbox_height={bbox_height:.1f}px known_height={known_height}m "
                            f"focal_length={format_debug_value(runtime_focal_length)}px "
                            f"raw={format_debug_value(raw_distance)}m "
                            f"smoothed={format_debug_value(smoothed_distance)}m"
                        )
                    if debug_motion:
                        print(
                            f"Motion debug: track_id={track_id} class={class_name} "
                            f"current={format_debug_value(motion.current_distance_m)}m "
                            f"previous={format_debug_value(motion.previous_distance_m)}m "
                            f"dt={format_debug_value(motion.dt_seconds)}s "
                            f"raw_closing={format_debug_value(motion.raw_closing_speed_mps)}m/s "
                            f"smoothed_closing={format_debug_value(motion.closing_speed_mps)}m/s "
                            f"state={motion.state} ttc={format_debug_value(motion.ttc_seconds)}s "
                            f"outlier_rejected={motion.outlier_rejected}"
                        )
                    if debug_risk:
                        print(
                            f"Risk debug: track_id={track_id} class={class_name} zone={path_zone} "
                            f"distance={format_debug_value(fused_distance)}m "
                            f"closing={format_debug_value(fused_closing)}m/s "
                            f"ttc={format_debug_value(fused_ttc)}s "
                            f"vehicle={speed_provider.get_vehicle_speed_kmh():.0f}km/h confidence={confidence:.2f} "
                            f"raw_risk={raw_risk.level} smoothed_risk={smoothed_level} "
                            f"score={raw_risk.score} reason={raw_risk.reason}"
                        )
                    if debug_sensors:
                        print(
                            f"Sensor debug: track_id={track_id} class={class_name} "
                            f"camera_bearing={format_debug_value(fused_state.camera_bearing_deg)}deg "
                            f"camera_distance={format_debug_value(fused_state.camera_distance_m)}m "
                            f"sensor_id={fused_state.sensor_object_id if fused_state.sensor_object_id is not None else 'N/A'} "
                            f"sensor_angle={format_debug_value(fused_state.sensor_angle_deg)}deg "
                            f"sensor_distance={format_debug_value(fused_state.sensor_distance_m)}m "
                            f"sensor_speed={format_debug_value(fused_state.sensor_relative_speed_mps)}m/s "
                            f"dt={format_debug_value(fused_state.timestamp_difference_s)}s "
                            f"association={fused_state.association_status}/{fused_state.fusion_confidence} "
                            f"fused_distance={format_debug_value(fused_state.fused_distance_m)}m[{fused_state.distance_source}] "
                            f"fused_closing={format_debug_value(fused_state.fused_closing_speed_mps)}m/s[{fused_state.closing_speed_source}]"
                        )

                track_history.cleanup_stale()
                distance_history.cleanup_stale()
                motion_analyzer.cleanup_stale()
                risk_smoother.cleanup_stale()

            display_now = time.perf_counter()
            elapsed = display_now - previous_display_time
            previous_display_time = display_now
            if elapsed > 0:
                fps = 1.0 / elapsed

            current_scene_risk = scene_risk(frame_risks)
            warning_state = warning_manager.update(warning_candidates, display_now)

            if display_now - last_console_log_time >= CONSOLE_LOG_INTERVAL_SECONDS:
                if no_detection:
                    print(f"Video FPS: {fps:.1f}" if fps is not None else "Video running")
                elif counts:
                    summary = ", ".join(f"{name}:{count}" for name, count in sorted(counts.items()))
                    print(
                        f"Tracked targets: {summary} | Tracks: {track_history.active_count()} | "
                        f"Inference: {inference_ms:.0f} ms | Enhance: {enhancement_ms:.1f} ms | Sensor: {sensor_mode_text} | Vehicle: {speed_provider.get_vehicle_speed_kmh():.0f} km/h | Scene Risk: {current_scene_risk} | Warning: {warning_state.level}"
                    )
                else:
                    print(
                        f"Tracked targets: none | Tracks: {track_history.active_count()} | "
                        f"Inference: {inference_ms:.0f} ms | Enhance: {enhancement_ms:.1f} ms | Sensor: {sensor_mode_text} | Vehicle: {speed_provider.get_vehicle_speed_kmh():.0f} km/h | Scene Risk: {current_scene_risk} | Warning: {warning_state.level}"
                        if inference_ms is not None
                        else "Tracked targets: none"
                    )
                last_console_log_time = display_now

            draw_header(
                frame,
                fps,
                inference_ms,
                source_name,
                no_detection,
                track_history.active_count(),
                distance_active,
                speed_provider.get_vehicle_speed_kmh(),
                speed_provider.source_name,
                current_scene_risk,
                warnings_enabled,
                audio_warning_enabled,
                visibility_metrics.visibility_class,
                enhancement_result.applied,
                enhancement_ms,
                sensor_mode_text,
                sensor_status,
            )
            if not no_detection:
                draw_warning_banner(frame, warning_state)
                draw_detection_summary(frame, counts)
                if debug_warning:
                    print(
                        f"Warning debug: scene_risk={current_scene_risk} threat_id={warning_state.track_id} "
                        f"class={warning_state.class_name} risk={warning_state.risk_level} "
                        f"distance={format_debug_value(warning_state.distance_m)}m "
                        f"ttc={format_debug_value(warning_state.ttc_seconds)}s "
                        f"previous={warning_state.previous_level} current={warning_state.level} "
                        f"audio_triggered={warning_state.audio_triggered} "
                        f"cooldown_remaining={warning_state.cooldown_remaining_seconds:.2f}s"
                    )
            display_frame = frame
            if show_enhancement:
                display_frame = cv2.hconcat([frame, enhancement_result.frame])
            cv2.imshow(WINDOW_TITLE, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                saved_path = save_detected_frame(frame, original_frame, enhancement_result.frame)
                print(f"Saved frame: {saved_path}")

    except KeyboardInterrupt:
        print()
        print("Detection interrupted by user.")
    except cv2.error as error:
        print(f"OpenCV error: {error}")
        return 1
    except Exception as error:
        print(f"Unexpected error during detection: {error}")
        return 1
    finally:
        capture.release()
        if sensor_source is not None:
            sensor_source.stop()
        cv2.destroyAllWindows()
        print("Camera stopped successfully.")

    return 0




























