"""Split an annotated YOLO source dataset into train, val, and test folders."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_IMAGES = PROJECT_ROOT / "dataset" / "annotated" / "images"
DEFAULT_SOURCE_LABELS = PROJECT_ROOT / "dataset" / "annotated" / "labels"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a manually annotated YOLO dataset into train/val/test folders."
    )
    parser.add_argument("--source-images", type=Path, default=DEFAULT_SOURCE_IMAGES)
    parser.add_argument("--source-labels", type=Path, default=DEFAULT_SOURCE_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--val", type=float, default=0.20, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.10, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="Copy or move image/label pairs. Copy is safer and is the default.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing existing output images or labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be split without writing files.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must add up to 1.0, got {total:.4f}.")


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def collect_items(source_images: Path, source_labels: Path) -> tuple[list[DatasetItem], list[Path]]:
    items: list[DatasetItem] = []
    missing_labels: list[Path] = []

    if not source_images.exists():
        raise FileNotFoundError(f"Source images folder does not exist: {source_images}")

    for image_path in sorted(source_images.rglob("*")):
        if not image_path.is_file() or image_path.name == ".gitkeep":
            continue
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        label_path = source_labels / image_path.relative_to(source_images).with_suffix(".txt")
        if not label_path.exists():
            missing_labels.append(image_path)
            continue

        items.append(DatasetItem(image_path=image_path, label_path=label_path))

    return items, missing_labels


def split_items(
    items: list[DatasetItem],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[DatasetItem]]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def ensure_output_dirs(output: Path) -> None:
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def check_overwrites(output: Path, split_map: dict[str, list[DatasetItem]], source_images: Path) -> list[Path]:
    conflicts: list[Path] = []
    for split, items in split_map.items():
        for item in items:
            relative_image = item.image_path.relative_to(source_images)
            relative_label = relative_image.with_suffix(".txt")
            image_target = output / "images" / split / relative_image
            label_target = output / "labels" / split / relative_label
            if image_target.exists():
                conflicts.append(image_target)
            if label_target.exists():
                conflicts.append(label_target)
    return conflicts


def transfer_items(
    output: Path,
    split_map: dict[str, list[DatasetItem]],
    source_images: Path,
    mode: str,
) -> None:
    operation = shutil.copy2 if mode == "copy" else shutil.move

    for split, items in split_map.items():
        for item in items:
            relative_image = item.image_path.relative_to(source_images)
            relative_label = relative_image.with_suffix(".txt")
            image_target = output / "images" / split / relative_image
            label_target = output / "labels" / split / relative_label

            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)

            operation(item.image_path, image_target)
            operation(item.label_path, label_target)


def print_summary(split_map: dict[str, list[DatasetItem]], missing_labels: list[Path]) -> None:
    print("Dataset split summary")
    print("=====================")
    for split in SPLITS:
        print(f"{split}: {len(split_map[split])} images")

    if missing_labels:
        print("")
        print(f"Skipped {len(missing_labels)} images with missing labels:")
        for path in missing_labels:
            print(f"- {relative_path(path)}")


def main() -> int:
    args = parse_args()
    validate_ratios(args.train, args.val, args.test)

    source_images = args.source_images.resolve()
    source_labels = args.source_labels.resolve()
    output = args.output.resolve()

    items, missing_labels = collect_items(source_images, source_labels)
    split_map = split_items(items, args.train, args.val, args.seed)

    conflicts = check_overwrites(output, split_map, source_images)
    if conflicts and not args.allow_overwrite:
        print("Refusing to overwrite existing output files.")
        print("Use --allow-overwrite only after reviewing these paths:")
        for path in conflicts:
            print(f"- {relative_path(path)}")
        return 2

    print_summary(split_map, missing_labels)

    if args.dry_run:
        print("")
        print("Dry run only. No files were copied or moved.")
        return 0

    ensure_output_dirs(output)
    transfer_items(output, split_map, source_images, args.mode)
    print("")
    print(f"Completed {args.mode} split into {relative_path(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
