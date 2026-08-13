"""Review saved Phase 6 prediction images one by one."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "results" / "phase6_predictions"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review saved Phase 6 prediction images.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def main() -> int:
    args = parse_args()
    images = image_files(args.predictions.resolve())
    if not images:
        print(f"No prediction images found in {args.predictions}")
        return 0

    index = 0
    print("Controls: N = next, P = previous, Q or Esc = quit")
    while True:
        image_path = images[index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            index = min(index + 1, len(images) - 1)
            continue

        title = f"Phase 6 Review {index + 1}/{len(images)} - {image_path.name}"
        cv2.imshow(title, image)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(title)

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P")):
            index = max(0, index - 1)
        else:
            index = min(len(images) - 1, index + 1)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
