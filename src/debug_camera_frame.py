"""Capture one frame from a camera source and run one YOLO prediction for debugging."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.detector import draw_detection, normalize_model_names, sanitize_camera_source


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results"
DEBUG_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11n.pt"
RAW_FRAME_PATH = OUTPUT_DIR / "phone_debug_frame.jpg"
ANNOTATED_FRAME_PATH = OUTPUT_DIR / "phone_debug_prediction.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one camera frame with YOLO.")
    parser.add_argument("--camera-source", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = sanitize_camera_source(args.camera_source)
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"opened: {capture.isOpened()}")
    if not capture.isOpened():
        return 1

    frame = None
    frames_read = 0
    start = time.perf_counter()
    while time.perf_counter() - start < 5:
        ok, image = capture.read()
        if ok and image is not None:
            frame = image
            frames_read += 1
    capture.release()

    print(f"frames_read: {frames_read}")
    print(f"frame_shape: {None if frame is None else frame.shape}")
    if frame is None:
        print("No valid frame received.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(RAW_FRAME_PATH), frame)

    model = YOLO(str(DEBUG_MODEL_PATH))
    result = model.predict(source=frame, conf=args.confidence, imgsz=args.imgsz, verbose=False)[0]
    names = normalize_model_names(model.names)
    annotated = frame.copy()

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = names.get(class_id, f"class_{class_id}")
        confidence = float(box.conf[0])
        detections.append((class_name, confidence))
        draw_detection(annotated, box.xyxy[0].tolist(), class_name, confidence)

    cv2.imwrite(str(ANNOTATED_FRAME_PATH), annotated)
    print("detections:")
    if detections:
        for class_name, confidence in detections:
            print(f"- {class_name}: {confidence:.3f}")
    else:
        print("- none")

    print(f"raw_frame: {RAW_FRAME_PATH}")
    print(f"annotated_frame: {ANNOTATED_FRAME_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

