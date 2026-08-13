"""Create a camera calibration file for Phase 09 distance estimation.

Distance calibration uses:

    focal_length_pixels = (pixel_height * known_distance) / known_height

The optional camera-assisted mode detects a visible object, displays its
bounding-box pixel height, and saves calibration when the user presses C.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from src.config import KNOWN_OBJECT_HEIGHTS_METERS, TARGET_CLASSES
from src.distance import DEFAULT_CALIBRATION_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "results" / "phase9_distance_report.txt"
CUSTOM_MODEL_PATH = PROJECT_ROOT / "models" / "airacare_animal_detector_best.pt"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"
WINDOW_TITLE = "Airacare Distance Calibration"
DEFAULT_CONFIDENCE = 0.25


def prompt_float(label: str) -> float:
    while True:
        raw_value = input(f"{label}: ").strip()
        try:
            value = float(raw_value)
        except ValueError:
            print("Enter a number.")
            continue
        if value <= 0:
            print("Value must be greater than 0.")
            continue
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate monocular distance estimation.")
    parser.add_argument("--known-height", type=float, help="Real object height in metres.")
    parser.add_argument("--known-distance", type=float, help="Measured distance from camera in metres.")
    parser.add_argument("--pixel-height", type=float, help="Detected object bounding-box height in pixels.")
    parser.add_argument("--camera-source", default="unknown", help="Camera index, phone stream URL, or profile name.")
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resolution used during manual calibration. Camera mode uses the actual frame size.",
    )
    parser.add_argument("--calibration-object", default="person", choices=TARGET_CLASSES, help="Object used for calibration.")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="Detection confidence for camera-assisted calibration.")
    parser.add_argument("--output", default=str(DEFAULT_CALIBRATION_PATH), help="Output calibration JSON path.")
    return parser.parse_args()


def sanitize_camera_source(source: str) -> str | int:
    source_text = str(source).strip().strip('"').strip("'")
    markdown_match = re.fullmatch(
        r"\[(?P<label>https?://[^\]]+|rtsp://[^\]]+)\]\((?P<url>https?://[^)]+|rtsp://[^)]+)\)",
        source_text,
    )
    if markdown_match:
        source_text = markdown_match.group("url")
    if source_text.isdigit():
        return int(source_text)
    if source_text.lower().startswith(("http://", "https://")) and source_text.rstrip("/").count("/") == 2:
        return source_text.rstrip("/") + "/video"
    return source_text


def model_class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return {int(k): str(v).lower() for k, v in names.items()}
    return {index: str(name).lower() for index, name in enumerate(names)}


def find_best_calibration_box(frame, model: YOLO, class_name: str, confidence: float) -> tuple[tuple[int, int, int, int], float] | None:
    results = model.predict(source=frame, conf=confidence, imgsz=640, verbose=False)
    result = results[0] if results else None
    boxes = getattr(result, "boxes", []) if result is not None else []
    names = model_class_names(model)
    best: tuple[tuple[int, int, int, int], float] | None = None
    best_confidence = -1.0
    for box in boxes:
        detected_class = names.get(int(box.cls[0]), "")
        if detected_class != class_name:
            continue
        box_confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        height = float(max(0, y2 - y1))
        if height <= 0 or box_confidence < best_confidence:
            continue
        best = ((x1, y1, x2, y2), height)
        best_confidence = box_confidence
    return best


def camera_assisted_pixel_height(args: argparse.Namespace) -> tuple[float, tuple[int, int]] | None:
    model_path = CUSTOM_MODEL_PATH if CUSTOM_MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    if not model_path.exists():
        print(f"Error: no YOLO model found for camera-assisted calibration: {model_path}")
        return None

    print(f"Loading calibration detector: {model_path}")
    model = YOLO(str(model_path))
    camera_source = sanitize_camera_source(str(args.camera_source))
    print(f"Opening calibration camera: {camera_source}")
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        print("Error: could not open calibration camera source.")
        return None

    try:
        print("Stand at the known distance with the calibration object fully visible.")
        print("Press C to save the shown box height. Press Q to quit.")
        selected_height: float | None = None
        frame_resolution = (1280, 720)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Error: could not read a calibration frame.")
                return None

            frame_height, frame_width = frame.shape[:2]
            frame_resolution = (frame_width, frame_height)
            match = find_best_calibration_box(frame, model, args.calibration_object, args.confidence)
            if match is not None:
                (x1, y1, x2, y2), pixel_height = match
                selected_height = pixel_height
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                label = f"{args.calibration_object.upper()} height={pixel_height:.1f}px - Press C"
                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Show one {args.calibration_object}. Press Q to quit.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                return None
            if key in (ord("c"), ord("C")) and selected_height is not None:
                return selected_height, frame_resolution
    finally:
        capture.release()
        cv2.destroyAllWindows()


def build_calibration(args: argparse.Namespace, pixel_height: float, resolution: tuple[int, int]) -> dict[str, Any]:
    known_height = args.known_height if args.known_height is not None else prompt_float("Known object height in metres")
    known_distance = args.known_distance if args.known_distance is not None else prompt_float("Known distance from camera in metres")

    if known_height <= 0 or known_distance <= 0 or pixel_height <= 0:
        raise ValueError("known height, known distance, and pixel height must be greater than 0")

    focal_length_pixels = (pixel_height * known_distance) / known_height
    return {
        "focal_length_pixels": focal_length_pixels,
        "camera_name": str(args.camera_source),
        "camera_source": str(args.camera_source),
        "calibration_width": int(resolution[0]),
        "calibration_height": int(resolution[1]),
        "calibration_resolution": [int(resolution[0]), int(resolution[1])],
        "known_object_class": args.calibration_object,
        "calibration_object": args.calibration_object,
        "known_object_height_m": known_height,
        "known_height_meters": known_height,
        "known_distance_m": known_distance,
        "known_distance_meters": known_distance,
        "pixel_height": pixel_height,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Distance is an estimate based on camera calibration and object size assumptions. Do not change zoom after calibration.",
    }


def write_report(calibration: dict, output_path: Path) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            [
                "Airacare Animal Detector",
                "Phase 09 Distance Calibration Report",
                "",
                f"Camera used: {calibration['camera_source']}",
                f"Calibration resolution: {calibration['calibration_width']}x{calibration['calibration_height']}",
                f"Focal length: {calibration['focal_length_pixels']:.2f} px",
                f"Object used for calibration: {calibration['known_object_class']}",
                f"Known object height: {calibration['known_object_height_m']:.2f} m",
                f"Known calibration distance: {calibration['known_distance_m']:.2f} m",
                f"Measured pixel height: {calibration['pixel_height']:.1f} px",
                f"Calibration file: {output_path}",
                "",
                "Physical distance test results:",
                "Actual distance | Estimated distance | Absolute error | Percentage error",
                "Fill this section after testing at known distances such as 2 m, 3 m, 5 m, and 8 m.",
                "",
                "Important limitation:",
                "Distance is an estimate based on camera calibration and object size assumptions.",
                "Animal sizes vary, so animal distances are approximate.",
                "Do not change camera zoom after calibration.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.pixel_height is not None:
        resolution = tuple(args.resolution or [1280, 720])
        pixel_height = args.pixel_height
    elif str(args.camera_source).lower() != "unknown":
        result = camera_assisted_pixel_height(args)
        if result is None:
            print("Calibration cancelled or failed; no file was written.")
            return 1
        pixel_height, resolution = result
    else:
        pixel_height = prompt_float("Object bounding-box height in pixels")
        resolution = tuple(args.resolution or [1280, 720])

    try:
        calibration = build_calibration(args, pixel_height, resolution)
    except ValueError as error:
        print(f"Error: {error}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(calibration, indent=4) + "\n", encoding="utf-8")
    write_report(calibration, output_path)

    print("Airacare distance calibration complete.")
    print(f"Focal length: {calibration['focal_length_pixels']:.2f} px")
    print(f"Resolution: {calibration['calibration_width']}x{calibration['calibration_height']}")
    print(f"Saved calibration: {output_path}")
    print(f"Saved report template: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
