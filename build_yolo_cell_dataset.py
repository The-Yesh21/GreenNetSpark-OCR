import argparse
import shutil
from pathlib import Path

import cv2

from yolo_cells import YOLO_CELL_CLASSES, detect_grid_cell_boxes, write_yolo_label


def iter_dataset_images(dataset_dir: Path) -> list[Path]:
    images: list[Path] = []
    for pattern in ("test*.jpg", "test*.jpeg", "test*.png"):
        images.extend(dataset_dir.glob(pattern))
    import re
    return sorted(images, key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)])


def write_data_yaml(output_dir: Path) -> None:
    names = ", ".join(f"'{name}'" for name in YOLO_CELL_CLASSES)
    content = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "\n"
        f"nc: {len(YOLO_CELL_CLASSES)}\n"
        f"names: [{names}]\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build initial YOLO labels for veg/price cell splitting.")
    parser.add_argument("--source", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("datasets/yolo_cell_split"))
    parser.add_argument("--val-every", type=int, default=5, help="Put every Nth image into validation")
    args = parser.parse_args()

    images = iter_dataset_images(args.source)
    if not images:
        raise SystemExit(f"No test*.jpg/jpeg/png images found in {args.source}")

    total_boxes = 0
    for idx, image_path in enumerate(images, start=1):
        split = "valid" if args.val_every > 0 and idx % args.val_every == 0 else "train"
        target_image = args.output / split / "images" / image_path.name
        target_label = args.output / split / "labels" / f"{image_path.stem}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes = detect_grid_cell_boxes(image_path)
        write_yolo_label(target_label, image.shape[:2], boxes)
        total_boxes += len(boxes)
        print(f"{image_path.name}: {len(boxes)} boxes -> {split}")

    write_data_yaml(args.output)
    print(f"Built {len(images)} images and {total_boxes} boxes -> {args.output}")
    print(f"YOLO data config: {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
