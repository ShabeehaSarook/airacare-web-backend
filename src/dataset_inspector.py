"""Inspect the raw Airacare custom dataset and write a quality report."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from src.config import TARGET_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RAW_DATASET_DIR = DATASET_ROOT / "raw"
REPORTS_DIR = DATASET_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "dataset_report.txt"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_WIDTH = 320
MIN_HEIGHT = 240
HASH_SIZE = 8


@dataclass
class ImageRecord:
    path: Path
    class_name: str
    width: int
    height: int
    exact_hash: str
    perceptual_hash: str


@dataclass
class DatasetInspection:
    class_counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in TARGET_CLASSES})
    records: list[ImageRecord] = field(default_factory=list)
    unsupported_files: list[Path] = field(default_factory=list)
    unreadable_images: list[Path] = field(default_factory=list)
    small_images: list[ImageRecord] = field(default_factory=list)
    exact_duplicates: dict[str, list[Path]] = field(default_factory=dict)
    near_duplicates: dict[str, list[Path]] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(self.class_counts.values())


def file_hash(path: Path) -> str:
    """Return a SHA-256 hash for exact duplicate detection."""
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(image) -> str:
    """Return a simple average hash for possible near-duplicate detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)
    average = resized.mean()
    bits = resized > average
    return "".join("1" if value else "0" for value in bits.flatten())


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def inspect_dataset() -> DatasetInspection:
    """Scan dataset/raw and collect image counts and quality warnings."""
    inspection = DatasetInspection()
    exact_hashes: dict[str, list[Path]] = defaultdict(list)
    perceptual_hashes: dict[str, list[Path]] = defaultdict(list)

    for class_name in TARGET_CLASSES:
        class_dir = RAW_DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep":
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                inspection.unsupported_files.append(path)
                continue

            image = cv2.imread(str(path))
            if image is None:
                inspection.unreadable_images.append(path)
                continue

            height, width = image.shape[:2]
            exact = file_hash(path)
            perceptual = average_hash(image)
            record = ImageRecord(path, class_name, width, height, exact, perceptual)

            inspection.class_counts[class_name] += 1
            inspection.records.append(record)
            exact_hashes[exact].append(path)
            perceptual_hashes[perceptual].append(path)

            if width < MIN_WIDTH or height < MIN_HEIGHT:
                inspection.small_images.append(record)

    inspection.exact_duplicates = {
        image_hash: paths for image_hash, paths in exact_hashes.items() if len(paths) > 1
    }
    inspection.near_duplicates = {
        image_hash: paths for image_hash, paths in perceptual_hashes.items() if len(paths) > 1
    }
    return inspection


def format_path_list(paths: list[Path]) -> list[str]:
    return [f"  - {relative_path(path)}" for path in paths]


def build_report(inspection: DatasetInspection) -> str:
    """Build the plain-text dataset inspection report."""
    lines: list[str] = ["Airacare Dataset Report", "========================", ""]

    lines.append("Image counts by class:")
    for class_name in TARGET_CLASSES:
        lines.append(f"{class_name}: {inspection.class_counts[class_name]} images")
    lines.append("")
    lines.append(f"Total: {inspection.total_images} images")
    lines.append("")

    if inspection.records:
        widths = [record.width for record in inspection.records]
        heights = [record.height for record in inspection.records]
        lines.append("Image dimensions:")
        lines.append(f"Minimum width: {min(widths)} px")
        lines.append(f"Minimum height: {min(heights)} px")
        lines.append(f"Maximum width: {max(widths)} px")
        lines.append(f"Maximum height: {max(heights)} px")
    else:
        lines.append("Image dimensions: no readable images found")
    lines.append("")

    lines.append("Warnings:")
    warnings_found = False

    if inspection.unreadable_images:
        warnings_found = True
        lines.append(f"- {len(inspection.unreadable_images)} unreadable/corrupted images")
        lines.extend(format_path_list(inspection.unreadable_images))

    if inspection.unsupported_files:
        warnings_found = True
        lines.append(f"- {len(inspection.unsupported_files)} unsupported files")
        lines.extend(format_path_list(inspection.unsupported_files))

    if inspection.small_images:
        warnings_found = True
        lines.append(
            f"- {len(inspection.small_images)} images below recommended resolution "
            f"({MIN_WIDTH}x{MIN_HEIGHT})"
        )
        for record in inspection.small_images:
            lines.append(f"  - {relative_path(record.path)} ({record.width}x{record.height})")

    exact_duplicate_count = sum(len(paths) for paths in inspection.exact_duplicates.values())
    if inspection.exact_duplicates:
        warnings_found = True
        lines.append(f"- {exact_duplicate_count} exact duplicate image files")
        for paths in inspection.exact_duplicates.values():
            lines.extend(format_path_list(paths))

    near_duplicate_count = sum(len(paths) for paths in inspection.near_duplicates.values())
    if inspection.near_duplicates:
        warnings_found = True
        lines.append(f"- {near_duplicate_count} possible near-duplicate image files")
        for paths in inspection.near_duplicates.values():
            lines.extend(format_path_list(paths))

    if not warnings_found:
        lines.append("- None")

    lines.append("")
    lines.append("Notes:")
    lines.append("- This script does not delete, move, rename, annotate, or train on source images.")
    lines.append("- Near-duplicate detection uses a simple average hash and should be manually reviewed.")
    lines.append("- Annotation and YOLO train/val/test splitting belong to later phases.")

    return "\n".join(lines) + "\n"


def write_report(report_text: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")


def main() -> int:
    inspection = inspect_dataset()
    report_text = build_report(inspection)
    write_report(report_text)
    print(report_text)
    print(f"Report written to: {relative_path(REPORT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())