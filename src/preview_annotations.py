"""Preview YOLO annotations by drawing boxes on a sample of dataset images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from src.config import TARGET_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "dataset"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
COLORS = [
    (46, 204, 113),
    (52, 152, 219),
    (231, 76, 60),
    (241, 196, 15),
    (155, 89, 182),
    (230, 126, 34),
    (26, 188, 156),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visually preview YOLO labels on images.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    return parser.parse_args()


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


def label_path_for(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    return labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")


def draw_annotations(image, label_path: Path) -> int:
    height, width = image.shape[:2]
    drawn = 0

    if not label_path.exists():
        cv2.putText(image, "missing label file", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return drawn

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = [float(value) for value in parts[1:]]
        except ValueError:
            continue

        if class_id < 0 or class_id >= len(TARGET_CLASSES):
            continue

        x1 = int((center_x - box_width / 2) * width)
        y1 = int((center_y - box_height / 2) * height)
        x2 = int((center_x + box_width / 2) * width)
        y2 = int((center_y + box_height / 2) * height)

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        color = COLORS[class_id % len(COLORS)]
        label = TARGET_CLASSES[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        drawn += 1

    return drawn


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    images_dir = dataset / "images" / args.split
    labels_dir = dataset / "labels" / args.split

    images = image_files(images_dir)
    if args.shuffle:
        random.Random(args.seed).shuffle(images)

    selected = images[: max(0, args.count)]
    if not selected:
        print(f"No images found in {images_dir}")
        return 0

    print("Press any key for next image. Press Q or Esc to quit.")
    for image_path in selected:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        label_path = label_path_for(image_path, images_dir, labels_dir)
        box_count = draw_annotations(image, label_path)
        window_title = f"{args.split}: {image_path.name} ({box_count} boxes)"
        cv2.imshow(window_title, image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_title)
        if key in (27, ord("q"), ord("Q")):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
