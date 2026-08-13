"""Compare original vs enhanced YOLO detections for Phase 14."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.detector import CUSTOM_MODEL_PATH, PRETRAINED_FALLBACK_MODEL_PATH
from src.image_enhancement import enhance_frame
from src.visibility import DEFAULT_VISIBILITY_CONFIG_PATH, analyze_visibility, load_visibility_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "images" / "test"
REPORT_PATH = PROJECT_ROOT / "results" / "phase14_visibility_report.txt"
FAILURE_DIR = PROJECT_ROOT / "results" / "phase14_failures"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO detection on original vs enhanced images.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR), help="Image file or folder to evaluate.")
    parser.add_argument("--model", default=None, help="Model path. Defaults to custom model if available, otherwise yolo11n.pt.")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--enhancement", choices=["off", "auto", "lowlight"], default="auto", help="Enhancement mode.")
    parser.add_argument("--visibility-config", default=str(DEFAULT_VISIBILITY_CONFIG_PATH), help="Visibility config JSON path.")
    parser.add_argument("--save-failures", action="store_true", help="Save examples where enhancement reduced accepted detections.")
    return parser.parse_args()


def iter_images(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        yield input_path
    elif input_path.is_dir():
        for path in sorted(input_path.rglob("*")):
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path


def count_detections(result) -> tuple[int, float]:
    boxes = getattr(result, "boxes", []) if result is not None else []
    count = len(boxes)
    if count == 0:
        return 0, 0.0
    confidences = [float(box.conf[0]) for box in boxes]
    return count, sum(confidences) / len(confidences)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model) if args.model else (CUSTOM_MODEL_PATH if CUSTOM_MODEL_PATH.exists() else PRETRAINED_FALLBACK_MODEL_PATH)
    if not model_path.exists():
        print(f"Error: model not found: {model_path}")
        return 1

    image_paths = list(iter_images(Path(args.input)))
    if not image_paths:
        print(f"No images found in {args.input}")
        return 1

    config = load_visibility_config(args.visibility_config)
    model = YOLO(str(model_path))
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Airacare Animal Detector",
        "Phase 14 Original vs Enhanced Evaluation",
        "",
        f"Model: {model_path}",
        f"Input: {Path(args.input)}",
        f"Enhancement mode: {args.enhancement}",
        f"Confidence: {args.confidence}",
        f"Image size: {args.imgsz}",
        "",
        "image | visibility | original_count | enhanced_count | original_avg_conf | enhanced_avg_conf | enhancement_ms | original_infer_ms | enhanced_infer_ms | notes",
    ]

    total_original = 0
    total_enhanced = 0
    total_enhancement_ms = 0.0

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            lines.append(f"{image_path.name} | unreadable | 0 | 0 | 0 | 0 | 0 | 0 | 0 | unreadable image")
            continue

        visibility = analyze_visibility(frame, config)
        enhancement = enhance_frame(frame, visibility.visibility_class, args.enhancement, config)

        start = time.perf_counter()
        original_results = model.predict(source=frame, conf=args.confidence, imgsz=args.imgsz, verbose=False)
        original_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        enhanced_results = model.predict(source=enhancement.frame, conf=args.confidence, imgsz=args.imgsz, verbose=False)
        enhanced_ms = (time.perf_counter() - start) * 1000

        original_count, original_conf = count_detections(original_results[0] if original_results else None)
        enhanced_count, enhanced_conf = count_detections(enhanced_results[0] if enhanced_results else None)
        total_original += original_count
        total_enhanced += enhanced_count
        total_enhancement_ms += enhancement.elapsed_ms

        notes = []
        if enhanced_count < original_count:
            notes.append("enhancement reduced accepted detections")
            if args.save_failures:
                cv2.imwrite(str(FAILURE_DIR / f"original_{image_path.name}"), frame)
                cv2.imwrite(str(FAILURE_DIR / f"enhanced_{image_path.name}"), enhancement.frame)
        elif enhanced_count > original_count:
            notes.append("enhancement increased accepted detections")
        if visibility.visibility_class == "VERY_DARK":
            notes.append("very dark frame")

        lines.append(
            f"{image_path.name} | {visibility.visibility_class} | {original_count} | {enhanced_count} | "
            f"{original_conf:.3f} | {enhanced_conf:.3f} | {enhancement.elapsed_ms:.1f} | "
            f"{original_ms:.1f} | {enhanced_ms:.1f} | {', '.join(notes) if notes else 'no note'}"
        )

    lines.extend(
        [
            "",
            f"Images evaluated: {len(image_paths)}",
            f"Total original detections: {total_original}",
            f"Total enhanced detections: {total_enhanced}",
            f"Average enhancement time: {total_enhancement_ms / len(image_paths):.1f} ms",
            "",
            "No precision/recall/mAP is reported unless labels/environment metadata are available.",
            "RGB enhancement is not thermal imaging and cannot recover details not captured by the camera.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluation complete: {REPORT_PATH}")
    print(f"Images evaluated: {len(image_paths)}")
    print(f"Original detections: {total_original}")
    print(f"Enhanced detections: {total_enhanced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
