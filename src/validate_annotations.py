"""Validate Airacare YOLO annotation files without modifying them."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.config import TARGET_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = {index: name for index, name in enumerate(TARGET_CLASSES)}


@dataclass
class ValidationReport:
    total_images: int = 0
    total_annotations: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)
    images_by_split: Counter[str] = field(default_factory=Counter)
    annotations_by_split: Counter[str] = field(default_factory=Counter)
    missing_label_files: list[Path] = field(default_factory=list)
    labels_with_missing_images: list[Path] = field(default_factory=list)
    empty_annotation_files: list[Path] = field(default_factory=list)
    malformed_lines: list[str] = field(default_factory=list)
    invalid_class_ids: list[str] = field(default_factory=list)
    invalid_coordinates: list[str] = field(default_factory=list)
    invalid_sizes: list[str] = field(default_factory=list)
    duplicate_label_lines: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return (
            len(self.missing_label_files)
            + len(self.labels_with_missing_images)
            + len(self.malformed_lines)
            + len(self.invalid_class_ids)
            + len(self.invalid_coordinates)
            + len(self.invalid_sizes)
        )

    @property
    def warning_count(self) -> int:
        return len(self.empty_annotation_files) + len(self.duplicate_label_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO annotation files for Airacare.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.name != ".gitkeep"
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def label_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*.txt") if path.is_file() and path.name != ".gitkeep")


def matching_image_paths(images_dir: Path, label_file: Path, labels_dir: Path) -> list[Path]:
    label_relative = label_file.relative_to(labels_dir)
    return [
        images_dir / label_relative.with_suffix(extension)
        for extension in SUPPORTED_IMAGE_EXTENSIONS
    ]


def validate_label_file(label_path: Path, split: str, report: ValidationReport) -> None:
    text = label_path.read_text(encoding="utf-8")
    raw_lines = text.splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    if not lines:
        report.empty_annotation_files.append(label_path)
        return

    seen_lines: set[str] = set()
    for line_number, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped in seen_lines:
            report.duplicate_label_lines.append(f"{relative_path(label_path)}:{line_number}: {stripped}")
        seen_lines.add(stripped)

        parts = stripped.split()
        if len(parts) != 5:
            report.malformed_lines.append(f"{relative_path(label_path)}:{line_number}: {stripped}")
            continue

        class_text, *coordinate_texts = parts
        try:
            class_id = int(class_text)
            values = [float(value) for value in coordinate_texts]
        except ValueError:
            report.malformed_lines.append(f"{relative_path(label_path)}:{line_number}: {stripped}")
            continue

        if class_id not in CLASS_NAMES:
            report.invalid_class_ids.append(f"{relative_path(label_path)}:{line_number}: {class_id}")
        else:
            report.class_counts[class_id] += 1
            report.total_annotations += 1
            report.annotations_by_split[split] += 1

        center_x, center_y, width, height = values
        if any(value < 0 or value > 1 for value in values):
            report.invalid_coordinates.append(f"{relative_path(label_path)}:{line_number}: {stripped}")

        if width <= 0 or height <= 0:
            report.invalid_sizes.append(f"{relative_path(label_path)}:{line_number}: {stripped}")


def validate_dataset(dataset: Path) -> ValidationReport:
    report = ValidationReport()

    for split in SPLITS:
        images_dir = dataset / "images" / split
        labels_dir = dataset / "labels" / split
        images = image_files(images_dir)
        labels = label_files(labels_dir)

        report.total_images += len(images)
        report.images_by_split[split] = len(images)

        label_lookup = {
            label.relative_to(labels_dir).with_suffix("").as_posix(): label
            for label in labels
        }
        image_lookup = defaultdict(list)
        for image in images:
            image_lookup[image.relative_to(images_dir).with_suffix("").as_posix()].append(image)

        for image in images:
            key = image.relative_to(images_dir).with_suffix("").as_posix()
            if key not in label_lookup:
                report.missing_label_files.append(image)

        for label in labels:
            if not any(path.exists() for path in matching_image_paths(images_dir, label, labels_dir)):
                report.labels_with_missing_images.append(label)
                continue
            validate_label_file(label, split, report)

    return report


def append_paths(lines: list[str], title: str, paths: list[Path]) -> None:
    lines.append(f"{title}: {len(paths)}")
    for path in paths:
        lines.append(f"- {relative_path(path)}")


def format_report(report: ValidationReport) -> str:
    lines: list[str] = ["Airacare YOLO Annotation Validation", "==================================", ""]

    lines.append("Image counts:")
    for split in SPLITS:
        lines.append(f"{split}: {report.images_by_split[split]} images")
    lines.append(f"total: {report.total_images} images")
    lines.append("")

    lines.append("Annotation counts:")
    for split in SPLITS:
        lines.append(f"{split}: {report.annotations_by_split[split]} boxes")
    lines.append(f"total: {report.total_annotations} boxes")
    lines.append("")

    lines.append("Bounding-box counts per class:")
    for class_id, class_name in CLASS_NAMES.items():
        lines.append(f"{class_name}: {report.class_counts[class_id]} objects")
    lines.append("")

    lines.append("Errors:")
    if report.error_count == 0:
        lines.append("- None")
    else:
        append_paths(lines, "Images with missing label files", report.missing_label_files)
        append_paths(lines, "Labels with missing image files", report.labels_with_missing_images)
        lines.append(f"Malformed lines: {len(report.malformed_lines)}")
        lines.extend(f"- {line}" for line in report.malformed_lines)
        lines.append(f"Invalid class IDs: {len(report.invalid_class_ids)}")
        lines.extend(f"- {line}" for line in report.invalid_class_ids)
        lines.append(f"Coordinates outside 0 to 1: {len(report.invalid_coordinates)}")
        lines.extend(f"- {line}" for line in report.invalid_coordinates)
        lines.append(f"Width or height <= 0: {len(report.invalid_sizes)}")
        lines.extend(f"- {line}" for line in report.invalid_sizes)
    lines.append("")

    lines.append("Warnings:")
    if report.warning_count == 0:
        lines.append("- None")
    else:
        append_paths(lines, "Empty annotation files", report.empty_annotation_files)
        lines.append(f"Duplicate label lines: {len(report.duplicate_label_lines)}")
        lines.extend(f"- {line}" for line in report.duplicate_label_lines)

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    report = validate_dataset(args.dataset.resolve())
    print(format_report(report))
    return 1 if report.error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
