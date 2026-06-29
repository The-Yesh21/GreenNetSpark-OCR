import argparse
from pathlib import Path

import cv2

from yolo_cells import YOLO_CELL_CLASSES


COLORS = {
    "left_veg": (0, 160, 0),
    "left_price": (180, 0, 180),
    "right_veg": (0, 120, 220),
    "right_price": (220, 120, 0),
}


def draw_labels(image_path: Path, label_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    h, w = image.shape[:2]

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls_id_s, cx_s, cy_s, bw_s, bh_s = line.split()[:5]
        label = YOLO_CELL_CLASSES[int(cls_id_s)]
        cx = float(cx_s) * w
        cy = float(cy_s) * h
        bw = float(bw_s) * w
        bh = float(bh_s) * h
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        color = COLORS[label]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO cell labels on an image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw_labels(args.image, args.label, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
