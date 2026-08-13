"""Train the Phase 5 custom Airacare YOLO detection model."""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import TARGET_CLASSES
from src.validate_annotations import format_report, image_files, label_files, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_YAML = DATASET_DIR / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_PROJECT = PROJECT_ROOT / "runs" / "airacare"
RUN_NAME = "phase5_training"
REPORT_PATH = RESULTS_DIR / "phase5_training_report.txt"
BEST_MODEL_COPY = MODELS_DIR / "airacare_animal_detector_best.pt"

BASE_MODEL = "yolo11n.pt"
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE: int | str = "auto"
DEVICE = "auto"


@dataclass
class TrainingSummary:
    base_model: str
    dataset: Path
    class_count: int
    epochs_requested: int
    image_size: int
    batch: int | str
    device_used: str
    duration_seconds: float | None = None
    best_model: Path | None = None
    last_model: Path | None = None
    copied_best_model: Path | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the custom Airacare YOLO model.")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="Path to Ultralytics data.yaml.")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Pretrained YOLO detection model.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch", default=BATCH_SIZE, help="Batch size or 'auto'.")
    parser.add_argument("--device", default=DEVICE, help="auto, cpu, cuda, 0, 0,1, or mps.")
    parser.add_argument("--project", type=Path, default=RUNS_PROJECT)
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--resume", action="store_true", help="Resume from the run's last.pt checkpoint.")
    parser.add_argument("--check-only", action="store_true", help="Run preflight checks without training.")
    parser.add_argument(
        "--allow-overwrite-model",
        action="store_true",
        help="Allow replacing models/airacare_animal_detector_best.pt after training.",
    )
    return parser.parse_args()


def parse_batch(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        batch = int(value)
    except ValueError as exc:
        raise ValueError("--batch must be an integer or 'auto'.") from exc
    if batch <= 0:
        raise ValueError("--batch must be greater than zero.")
    return batch


def parse_data_yaml_names(data_yaml: Path) -> list[str]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing dataset config: {data_yaml}")

    names: dict[int, str] = {}
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if line.strip() == "names:":
            in_names = True
            continue

        if in_names:
            if not raw_line.startswith((" ", "\t")):
                break
            key, separator, value = line.strip().partition(":")
            if not separator:
                continue
            try:
                class_id = int(key)
            except ValueError:
                continue
            names[class_id] = value.strip().strip("'\"")

    return [names[index] for index in sorted(names)]


def resolve_dataset_dir(data_yaml: Path) -> Path:
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line.startswith("path:"):
            continue
        value = line.partition(":")[2].strip().strip("'\"")
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return data_yaml.parent.resolve()


def resolve_device(requested: str) -> str:
    requested = str(requested).strip().lower()
    if requested == "auto":
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

    if requested in {"cpu", "mps"}:
        if requested == "mps":
            import torch

            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("Requested Apple MPS device, but MPS is not available.")
        return requested

    if requested.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return "0" if requested == "cuda" else requested.replace("cuda:", "")

    if all(part.strip().isdigit() for part in requested.split(",")):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA GPU index, but CUDA is not available.")
        return requested

    raise ValueError("--device must be one of: auto, cpu, cuda, cuda:0, 0, 0,1, or mps.")


def preflight_dataset(data_yaml: Path) -> tuple[Path, str]:
    data_yaml = data_yaml.resolve()
    names = parse_data_yaml_names(data_yaml)
    if names != TARGET_CLASSES:
        raise RuntimeError(
            "dataset/data.yaml must contain exactly these classes in order: "
            + ", ".join(TARGET_CLASSES)
        )

    dataset_dir = resolve_dataset_dir(data_yaml)
    train_images = dataset_dir / "images" / "train"
    val_images = dataset_dir / "images" / "val"
    train_labels = dataset_dir / "labels" / "train"
    val_labels = dataset_dir / "labels" / "val"

    required_dirs = (train_images, val_images, train_labels, val_labels)
    missing_dirs = [path for path in required_dirs if not path.exists()]
    if missing_dirs:
        joined = "\n".join(f"- {path}" for path in missing_dirs)
        raise RuntimeError(f"Missing required dataset folders:\n{joined}")

    if not image_files(train_images):
        raise RuntimeError(f"No training images found in {train_images}")
    if not image_files(val_images):
        raise RuntimeError(f"No validation images found in {val_images}")
    if not label_files(train_labels):
        raise RuntimeError(f"No training labels found in {train_labels}")
    if not label_files(val_labels):
        raise RuntimeError(f"No validation labels found in {val_labels}")

    validation_report = validate_dataset(dataset_dir)
    if validation_report.error_count:
        raise RuntimeError(
            "Annotation validation failed. Fix these errors before training:\n\n"
            + format_report(validation_report)
        )
    if validation_report.total_annotations == 0:
        raise RuntimeError("Dataset contains no bounding-box annotations.")

    return dataset_dir, format_report(validation_report)


def import_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Could not import Ultralytics YOLO. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return YOLO


def run_validation(model: Any, data_yaml: Path, device: str) -> dict[str, Any]:
    metrics = model.val(data=str(data_yaml), device=device)
    box = getattr(metrics, "box", None)
    if box is None:
        return {}

    results: dict[str, Any] = {
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
        "map50": getattr(box, "map50", None),
        "map50_95": getattr(box, "map", None),
        "per_class": {},
    }

    maps = getattr(box, "maps", None)
    if maps is not None:
        for index, class_name in enumerate(TARGET_CLASSES):
            if index < len(maps):
                results["per_class"][class_name] = {"map50_95": float(maps[index])}

    return results


def write_training_report(summary: TrainingSummary) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = summary.metrics
    per_class = metrics.get("per_class", {}) if metrics else {}

    lines = [
        "Airacare Animal Detector",
        "Phase 5 Training Report",
        "",
        f"Base model: {summary.base_model}",
        f"Dataset: {summary.dataset}",
        f"Number of classes: {summary.class_count}",
        f"Epochs: {summary.epochs_requested}",
        f"Image size: {summary.image_size}",
        f"Batch: {summary.batch}",
        f"Device used: {summary.device_used}",
        f"Training duration: {format_duration(summary.duration_seconds)}",
        "",
        "Overall metrics:",
        f"Precision: {format_metric(metrics.get('precision'))}",
        f"Recall: {format_metric(metrics.get('recall'))}",
        f"mAP50: {format_metric(metrics.get('map50'))}",
        f"mAP50-95: {format_metric(metrics.get('map50_95'))}",
        "",
        "Per-class results where available:",
    ]

    for class_name in TARGET_CLASSES:
        class_metrics = per_class.get(class_name, {})
        lines.append(f"{class_name}: mAP50-95={format_metric(class_metrics.get('map50_95'))}")

    lines.extend(
        [
            "",
            f"Best model location: {summary.best_model or 'not available'}",
            f"Last model location: {summary.last_model or 'not available'}",
            f"Copied best model: {summary.copied_best_model or 'not copied'}",
            "",
            "Warnings or issues:",
        ]
    )

    if summary.warnings or summary.issues:
        lines.extend(f"- {message}" for message in [*summary.warnings, *summary.issues])
    else:
        lines.append("- None")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: Any) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "not available"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def checkpoint_path(project: Path, name: str, checkpoint: str) -> Path:
    return project.resolve() / name / "weights" / checkpoint


def copy_best_model(best_model: Path, allow_overwrite: bool) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_MODEL_COPY.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing model: {BEST_MODEL_COPY}. "
            "Use --allow-overwrite-model after reviewing the current file."
        )
    shutil.copy2(best_model, BEST_MODEL_COPY)
    return BEST_MODEL_COPY


def main() -> int:
    args = parse_args()

    try:
        batch = parse_batch(str(args.batch))
        device = resolve_device(args.device)
        dataset_dir, validation_text = preflight_dataset(args.data)
    except Exception as exc:
        print(f"Preflight failed: {exc}")
        return 2

    args.project.mkdir(parents=True, exist_ok=True)
    print(validation_text)
    print(f"Device selected: {device}")
    print(f"Base model: {args.base_model}")
    print(f"Run output: {args.project.resolve() / args.name}")

    if args.check_only:
        print("Check-only mode complete. Training was not started.")
        return 0

    summary = TrainingSummary(
        base_model=args.base_model,
        dataset=dataset_dir,
        class_count=len(TARGET_CLASSES),
        epochs_requested=args.epochs,
        image_size=args.imgsz,
        batch=batch,
        device_used=device,
    )

    try:
        YOLO = import_yolo()
        model_source = checkpoint_path(args.project, args.name, "last.pt") if args.resume else args.base_model
        if args.resume and not model_source.exists():
            raise FileNotFoundError(f"Cannot resume because checkpoint does not exist: {model_source}")
        run_dir = args.project.resolve() / args.name
        if run_dir.exists() and not args.resume:
            raise FileExistsError(
                f"Training run directory already exists: {run_dir}. "
                "Use --resume for an interrupted run, or choose a different --name."
            )

        model = YOLO(str(model_source))
        start = time.perf_counter()
        model.train(
            data=str(args.data.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            device=device,
            project=str(args.project.resolve()),
            name=args.name,
            resume=args.resume,
            exist_ok=False,
        )
        summary.duration_seconds = time.perf_counter() - start

        summary.best_model = checkpoint_path(args.project, args.name, "best.pt")
        summary.last_model = checkpoint_path(args.project, args.name, "last.pt")
        if not summary.best_model.exists():
            raise FileNotFoundError(f"Training finished, but best.pt was not found: {summary.best_model}")
        if not summary.last_model.exists():
            summary.warnings.append(f"last.pt was not found: {summary.last_model}")

        summary.metrics = run_validation(YOLO(str(summary.best_model)), args.data.resolve(), device)
        summary.copied_best_model = copy_best_model(summary.best_model, args.allow_overwrite_model)
        write_training_report(summary)

        print(f"best.pt: {summary.best_model}")
        print(f"last.pt: {summary.last_model}")
        print(f"Copied best model to: {summary.copied_best_model}")
        print(f"Training report: {REPORT_PATH}")
        return 0

    except KeyboardInterrupt:
        summary.issues.append("Training was interrupted by the user.")
        write_training_report(summary)
        print("Training interrupted. Resume later with: python -m src.train_model --resume")
        return 130
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower() or "cuda" in message.lower() and "memory" in message.lower():
            message += "\nCUDA memory issue: retry with --batch 4, --batch 2, or --device cpu."
        summary.issues.append(message)
        write_training_report(summary)
        print(f"Training failed: {message}")
        return 1
    except Exception as exc:
        summary.issues.append(str(exc))
        write_training_report(summary)
        print(f"Training failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


