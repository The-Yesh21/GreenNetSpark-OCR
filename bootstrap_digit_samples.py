import argparse
import json
from pathlib import Path

import cv2

from handwritten_digits import segment_digits


def main() -> None:
    parser = argparse.ArgumentParser(description="Create labelled digit crops from confident OCR price rows.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--debug-json", type=Path, default=Path("ocr_debug.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("digit_samples"))
    parser.add_argument("--min-price-confidence", type=float, default=0.90)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)

    payload = json.loads(args.debug_json.read_text(encoding="utf-8"))
    saved = 0
    skipped = 0
    for row_index, row in enumerate(payload.get("rows", []), start=1):
        price = str(row.get("price", ""))
        if not price.isdigit():
            skipped += 1
            continue
        if float(row.get("price_confidence", 0.0)) < args.min_price_confidence:
            skipped += 1
            continue
        box = row.get("price_box") or []
        if len(box) != 4:
            skipped += 1
            continue
        x1, y1, x2, y2 = map(int, box)
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        segments = segment_digits(crop)
        if len(segments) != len(price):
            skipped += 1
            continue

        for digit_index, ((_, seg_box), digit) in enumerate(zip(segments, price), start=1):
            sx1, sy1, sx2, sy2 = seg_box
            digit_crop = crop[sy1:sy2, sx1:sx2]
            out_dir = args.output_dir / digit
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{args.image.stem}_row{row_index:02d}_{digit_index}.png"
            cv2.imwrite(str(out_path), digit_crop)
            saved += 1

    print(f"Saved {saved} digit crops into {args.output_dir}; skipped {skipped} rows")


if __name__ == "__main__":
    main()
