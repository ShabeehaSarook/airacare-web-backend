"""Evaluate the trained Airacare YOLO model on the Phase 4 test split."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from src.config import TARGET_CLASSES
from src.train_model import (
    BEST_MODEL_COPY,
    DATA_YAML,
    PROJECT_ROOT,
    format_metric,
    import_yolo,
    parse_data_yaml_names,
    resolve_dataset_dir,
    resolve_device,
)
from src.validate_annotations import format_report, image_files, label_files, validate_dataset


RESULTS_DIR = PROJECT_ROOT / "results"
EVALUATION_DIR = RESULTS_DIR / "phase6_evaluation"
PREDICTIONS_DIR = RESULTS_DIR / "phase6_predictions"
REPORT_PATH = RESULTS_DIR / "phase6_evaluation_report.txt"
DEFAULT_MODEL = BEST_MODEL_COPY
CONFIDENCE_THRESHOLDS = (0.30, 0.50, 0.70)
IOU_MATCH_THRESHOLD = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.35
HIGH_CONFIDENCE_THRESHOLD = 0.70
SIZE_BUCKETS = {
    "small/far": (0.0, 0.03),
    "medium": (0.03, 0.15),
    "large/near": (0.15, 1.01),
}


@dataclass
class Detection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass
class GroundTruth:
    class_id: int
    xyxy: tuple[float, float, float, float]
    area_ratio: float


@dataclass
class EvaluationSummary:
    model_path: Path
    dataset_path: Path
    device: str
    metrics: dict[str, Any] = field(default_factory=dict)
    threshold_counts: dict[float, int] = field(default_factory=dict)
    false_positives: list[str] = field(default_factory=list)
    false_negatives: list[str] = field(default_factory=list)
    class_confusions: list[str] = field(default_factory=list)
    low_confidence_detections: list[str] = field(default_factory=list)
    size_analysis: dict[str, dict[str, int]] = field(default_factory=dict)
    average_inference_ms: float | None = None
    fps: float | None = None
    weakest_class: str | None = None
    strongest_class: str | None = None
    recommended_threshold: str = "not available"
    readiness: str = "MODEL NEEDS IMPROVEMENT"
    warnings: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained Airacare YOLO model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, 0, 0,1, or mps.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence used for saved prediction images.")
    parser.add_argument("--iou", type=float, default=IOU_MATCH_THRESHOLD)
    parser.add_argument("--max-images", type=int, default=0, help="Limit prediction review images; 0 means all.")
    parser.add_argument("--check-only", action="store_true", help="Run preflight checks without evaluation.")
    parser.add_argument(
        "--allow-overwrite-results",
        action="store_true",
        help="Allow writing into existing Phase 6 result folders.",
    )
    return parser.parse_args()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def file_signature(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_output_dir(path: Path, allow_overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Output folder already contains files: {path}. "
            "Use --allow-overwrite-results after reviewing existing results."
        )
    path.mkdir(parents=True, exist_ok=True)


def preflight(args: argparse.Namespace) -> tuple[Any, Path, list[Path], str]:
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset config: {data_yaml}")

    names = parse_data_yaml_names(data_yaml)
    if names != TARGET_CLASSES:
        raise RuntimeError("dataset/data.yaml must contain exactly: " + ", ".join(TARGET_CLASSES))

    dataset_path = resolve_dataset_dir(data_yaml)
    test_images_dir = dataset_path / "images" / "test"
    test_labels_dir = dataset_path / "labels" / "test"
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Missing test images folder: {test_images_dir}")
    if not test_labels_dir.exists():
        raise FileNotFoundError(f"Missing test labels folder: {test_labels_dir}")

    test_images = image_files(test_images_dir)
    test_labels = label_files(test_labels_dir)
    if not test_images:
        raise RuntimeError(f"No test images found in {test_images_dir}")
    if not test_labels:
        raise RuntimeError(f"No test labels found in {test_labels_dir}")

    annotation_report = validate_dataset(dataset_path)
    if annotation_report.error_count:
        raise RuntimeError("Annotation validation failed:\n\n" + format_report(annotation_report))

    YOLO = import_yolo()
    model = YOLO(str(model_path))
    model_names = normalize_model_names(getattr(model, "names", {}))
    if model_names != TARGET_CLASSES:
        raise RuntimeError(
            "The model class names do not match the expected 7 classes. "
            f"Expected {TARGET_CLASSES}, got {model_names}."
        )

    return model, dataset_path, test_images, format_report(annotation_report)


def normalize_model_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def extract_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {}

    per_class: dict[str, dict[str, Any]] = {}
    class_precision = safe_sequence(getattr(box, "p", None))
    class_recall = safe_sequence(getattr(box, "r", None))
    class_map50 = safe_sequence(getattr(box, "ap50", None))
    class_map = safe_sequence(getattr(box, "maps", None))

    for index, class_name in enumerate(TARGET_CLASSES):
        per_class[class_name] = {
            "precision": class_precision[index] if index < len(class_precision) else None,
            "recall": class_recall[index] if index < len(class_recall) else None,
            "map50": class_map50[index] if index < len(class_map50) else None,
            "map50_95": class_map[index] if index < len(class_map) else None,
        }

    return {
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
        "map50": getattr(box, "map50", None),
        "map50_95": getattr(box, "map", None),
        "per_class": per_class,
    }


def safe_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        return [float(item) for item in value]
    except TypeError:
        return []


def read_ground_truth(label_path: Path, image_width: int, image_height: int) -> list[GroundTruth]:
    if not label_path.exists():
        return []

    boxes: list[GroundTruth] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        center_x, center_y, width, height = [float(value) for value in parts[1:]]
        x1 = (center_x - width / 2) * image_width
        y1 = (center_y - height / 2) * image_height
        x2 = (center_x + width / 2) * image_width
        y2 = (center_y + height / 2) * image_height
        boxes.append(GroundTruth(class_id, (x1, y1, x2, y2), width * height))
    return boxes


def detections_from_result(result: Any) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    detections: list[Detection] = []
    for box in boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        xyxy_values = tuple(float(value) for value in box.xyxy[0].tolist())
        detections.append(Detection(class_id, confidence, xyxy_values))
    return detections


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def draw_prediction(image: Any, detections: list[Detection]) -> Any:
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.xyxy]
        class_name = TARGET_CLASSES[detection.class_id] if detection.class_id < len(TARGET_CLASSES) else str(detection.class_id)
        color = (40, 180, 90)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{class_name} {detection.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return image


def analyze_predictions(
    model: Any,
    test_images: list[Path],
    dataset_path: Path,
    device: str,
    imgsz: int,
    conf: float,
    iou_threshold: float,
    max_images: int,
) -> tuple[list[Path], dict[str, Any]]:
    selected_images = test_images if max_images <= 0 else test_images[:max_images]
    labels_dir = dataset_path / "labels" / "test"
    images_dir = dataset_path / "images" / "test"
    saved_images: list[Path] = []
    inference_times: list[float] = []
    analysis: dict[str, Any] = {
        "false_positives": [],
        "false_negatives": [],
        "class_confusions": [],
        "low_confidence_detections": [],
        "size_analysis": {name: {"total": 0, "matched": 0} for name in SIZE_BUCKETS},
        "threshold_counts": {},
    }

    for image_path in selected_images:
        image = cv2.imread(str(image_path))
        if image is None:
            analysis["false_negatives"].append(f"{relative_path(image_path)}: unreadable test image")
            continue

        height, width = image.shape[:2]
        label_path = labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")
        ground_truths = read_ground_truth(label_path, width, height)

        start = time.perf_counter()
        results = model.predict(str(image_path), imgsz=imgsz, conf=conf, device=device, verbose=False)
        inference_times.append((time.perf_counter() - start) * 1000)

        detections = detections_from_result(results[0]) if results else []
        save_path = PREDICTIONS_DIR / image_path.name
        annotated = draw_prediction(image.copy(), detections)
        cv2.imwrite(str(save_path), annotated)
        saved_images.append(save_path)

        for threshold in CONFIDENCE_THRESHOLDS:
            analysis["threshold_counts"][threshold] = analysis["threshold_counts"].get(threshold, 0) + sum(
                1 for detection in detections if detection.confidence >= threshold
            )

        matched_detection_indexes: set[int] = set()
        for truth in ground_truths:
            bucket = size_bucket(truth.area_ratio)
            analysis["size_analysis"][bucket]["total"] += 1
            best_index = None
            best_iou = 0.0
            for index, detection in enumerate(detections):
                score = iou(truth.xyxy, detection.xyxy)
                if score > best_iou:
                    best_iou = score
                    best_index = index

            if best_index is None or best_iou < iou_threshold:
                analysis["false_negatives"].append(
                    f"{relative_path(image_path)}: missed {TARGET_CLASSES[truth.class_id]}"
                )
                continue

            matched_detection_indexes.add(best_index)
            analysis["size_analysis"][bucket]["matched"] += 1
            detection = detections[best_index]
            if detection.class_id != truth.class_id:
                analysis["class_confusions"].append(
                    f"{relative_path(image_path)}: {TARGET_CLASSES[truth.class_id]} predicted as "
                    f"{TARGET_CLASSES[detection.class_id]} ({detection.confidence:.2f})"
                )

        for index, detection in enumerate(detections):
            class_name = TARGET_CLASSES[detection.class_id]
            if detection.confidence < LOW_CONFIDENCE_THRESHOLD:
                analysis["low_confidence_detections"].append(
                    f"{relative_path(image_path)}: {class_name} {detection.confidence:.2f}"
                )
            if index not in matched_detection_indexes and detection.confidence >= HIGH_CONFIDENCE_THRESHOLD:
                analysis["false_positives"].append(
                    f"{relative_path(image_path)}: {class_name} {detection.confidence:.2f}"
                )

    if inference_times:
        average = sum(inference_times) / len(inference_times)
        analysis["average_inference_ms"] = average
        analysis["fps"] = 1000 / average if average > 0 else None

    return saved_images, analysis


def size_bucket(area_ratio: float) -> str:
    for name, (minimum, maximum) in SIZE_BUCKETS.items():
        if minimum <= area_ratio < maximum:
            return name
    return "large/near"


def choose_strengths(per_class: dict[str, dict[str, Any]]) -> tuple[str | None, str | None]:
    values: list[tuple[str, float]] = []
    for class_name, metrics in per_class.items():
        value = metrics.get("map50_95")
        if value is not None:
            values.append((class_name, float(value)))
    if not values:
        return None, None
    values.sort(key=lambda item: item[1])
    return values[0][0], values[-1][0]


def recommend_threshold(threshold_counts: dict[float, int]) -> str:
    if not threshold_counts:
        return "not available"
    return "Review 0.30, 0.50, and 0.70 saved count tradeoffs before selecting a deployment threshold."


def readiness(summary: EvaluationSummary) -> str:
    metrics = summary.metrics
    required = (metrics.get("precision"), metrics.get("recall"), metrics.get("map50"), metrics.get("map50_95"))
    if any(value is None for value in required):
        return "MODEL NEEDS IMPROVEMENT"
    if float(metrics["precision"]) >= 0.70 and float(metrics["recall"]) >= 0.70 and float(metrics["map50"]) >= 0.70:
        return "READY FOR PHASE 7"
    return "MODEL NEEDS IMPROVEMENT"


def write_report(summary: EvaluationSummary) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_class = summary.metrics.get("per_class", {}) if summary.metrics else {}
    lines = [
        "Airacare Animal Detector",
        "Phase 6 Evaluation Report",
        "",
        f"Model: {summary.model_path}",
        f"Dataset: {summary.dataset_path}",
        f"Device: {summary.device}",
        "",
        "Overall:",
        f"Precision: {format_metric(summary.metrics.get('precision'))}",
        f"Recall: {format_metric(summary.metrics.get('recall'))}",
        f"mAP50: {format_metric(summary.metrics.get('map50'))}",
        f"mAP50-95: {format_metric(summary.metrics.get('map50_95'))}",
        "",
        "Per-class metrics:",
    ]

    for class_name in TARGET_CLASSES:
        class_metrics = per_class.get(class_name, {})
        lines.extend(
            [
                f"{class_name.title()}:",
                f"Precision: {format_metric(class_metrics.get('precision'))}",
                f"Recall: {format_metric(class_metrics.get('recall'))}",
                f"mAP50: {format_metric(class_metrics.get('map50'))}",
                f"mAP50-95: {format_metric(class_metrics.get('map50_95'))}",
                "",
            ]
        )

    lines.extend(
        [
            "Inference:",
            f"Average inference time: {format_ms(summary.average_inference_ms)}",
            f"Approx FPS: {format_metric(summary.fps)}",
            "",
            f"Weakest class: {summary.weakest_class or 'not available'}",
            f"Strongest class: {summary.strongest_class or 'not available'}",
            "",
            "Confidence threshold test:",
        ]
    )

    if summary.threshold_counts:
        for threshold in CONFIDENCE_THRESHOLDS:
            lines.append(f"{threshold:.2f}: {summary.threshold_counts.get(threshold, 0)} detections")
    else:
        lines.append("not available")

    lines.extend(
        [
            "",
            "Image-size-based distance proxy:",
            "This is based on bounding-box area in the image, not real meter measurement.",
        ]
    )
    if summary.size_analysis:
        for bucket, values in summary.size_analysis.items():
            total = values.get("total", 0)
            matched = values.get("matched", 0)
            rate = matched / total if total else None
            lines.append(f"{bucket}: {matched}/{total} matched, rate={format_metric(rate)}")
    else:
        lines.append("not available")

    lines.extend(
        [
            "",
            "Lighting condition review:",
            "Condition-specific analysis requires manual grouping or metadata; no condition labels were found in the current dataset structure.",
            "",
            "Common errors:",
        ]
    )
    if summary.class_confusions:
        lines.extend(f"- {item}" for item in summary.class_confusions[:25])
    else:
        lines.append("- None found in automated IoU review.")

    lines.append("")
    lines.append("False positives:")
    if summary.false_positives:
        lines.extend(f"- {item}" for item in summary.false_positives[:25])
    else:
        lines.append("- None found in automated IoU review.")

    lines.append("")
    lines.append("False negatives:")
    if summary.false_negatives:
        lines.extend(f"- {item}" for item in summary.false_negatives[:25])
    else:
        lines.append("- None found in automated IoU review.")

    lines.extend(
        [
            "",
            f"Recommended confidence threshold: {summary.recommended_threshold}",
            f"Readiness: {summary.readiness}",
            "",
            "Warnings or issues:",
        ]
    )
    if summary.warnings:
        lines.extend(f"- {warning}" for warning in summary.warnings)
    else:
        lines.append("- None")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_ms(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.3f} ms"


def copy_confusion_matrix() -> None:
    source = EVALUATION_DIR / "confusion_matrix.png"
    if source.exists():
        return
    for candidate in EVALUATION_DIR.rglob("confusion_matrix*.png"):
        if candidate != source:
            shutil.copy2(candidate, source)
            return


def main() -> int:
    args = parse_args()

    try:
        device = resolve_device(args.device)
        model, dataset_path, test_images, annotation_text = preflight(args)
        if not args.check_only:
            verify_output_dir(EVALUATION_DIR, args.allow_overwrite_results)
            verify_output_dir(PREDICTIONS_DIR, args.allow_overwrite_results)
    except Exception as exc:
        print(f"Evaluation preflight failed: {exc}")
        return 2

    print(annotation_text)
    print(f"Model: {args.model.resolve()}")
    print(f"Dataset: {dataset_path}")
    print(f"Device selected: {device}")

    if args.check_only:
        print("Check-only mode complete. Evaluation was not started.")
        return 0

    signatures_before = {path: file_signature(path) for path in test_images}

    try:
        metrics_result = model.val(
            data=str(args.data.resolve()),
            split="test",
            imgsz=args.imgsz,
            device=device,
            project=str(RESULTS_DIR.resolve()),
            name="phase6_evaluation",
            exist_ok=True,
        )
        metrics = extract_metrics(metrics_result)

        saved_images, analysis = analyze_predictions(
            model=model,
            test_images=test_images,
            dataset_path=dataset_path,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou_threshold=args.iou,
            max_images=args.max_images,
        )

        signatures_after = {path: file_signature(path) for path in test_images}
        unchanged = signatures_before == signatures_after
        warnings = []
        if not unchanged:
            warnings.append("One or more source test images changed during evaluation.")
        if not saved_images:
            warnings.append("No prediction images were saved.")

        weakest, strongest = choose_strengths(metrics.get("per_class", {}) if metrics else {})
        summary = EvaluationSummary(
            model_path=args.model.resolve(),
            dataset_path=dataset_path,
            device=device,
            metrics=metrics,
            threshold_counts=analysis.get("threshold_counts", {}),
            false_positives=analysis.get("false_positives", []),
            false_negatives=analysis.get("false_negatives", []),
            class_confusions=analysis.get("class_confusions", []),
            low_confidence_detections=analysis.get("low_confidence_detections", []),
            size_analysis=analysis.get("size_analysis", {}),
            average_inference_ms=analysis.get("average_inference_ms"),
            fps=analysis.get("fps"),
            weakest_class=weakest,
            strongest_class=strongest,
            recommended_threshold=recommend_threshold(analysis.get("threshold_counts", {})),
            warnings=warnings,
        )
        summary.readiness = readiness(summary)

        copy_confusion_matrix()
        write_report(summary)

        print(f"Evaluation visuals: {EVALUATION_DIR}")
        print(f"Prediction images: {PREDICTIONS_DIR}")
        print(f"Evaluation report: {REPORT_PATH}")
        print(f"Precision: {format_metric(metrics.get('precision'))}")
        print(f"Recall: {format_metric(metrics.get('recall'))}")
        print(f"mAP50: {format_metric(metrics.get('map50'))}")
        print(f"mAP50-95: {format_metric(metrics.get('map50_95'))}")
        print(f"Average inference: {format_ms(summary.average_inference_ms)}")
        print(f"Approx FPS: {format_metric(summary.fps)}")
        print(f"Readiness: {summary.readiness}")
        return 0

    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

