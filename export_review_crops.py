import argparse
import csv
import json
from pathlib import Path

import cv2


def crop_box(image, box: list[float], pad: int = 6):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]


def should_export(row: dict, include: str) -> bool:
    if include == "all":
        return True
    if include == "pending-price":
        return row.get("Price") == "PENDING_REVIEW"
    if include == "pending-name":
        return row.get("Vegetable") == "PENDING_REVIEW"
    if include == "pending-any":
        return row.get("Price") == "PENDING_REVIEW" or row.get("Vegetable") == "PENDING_REVIEW"
    raise ValueError(f"Unknown include mode: {include}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO split OCR review crops from debug JSON.")
    parser.add_argument("--debug-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("review_crops"))
    parser.add_argument(
        "--include",
        choices=["pending-price", "pending-name", "pending-any", "all"],
        default="pending-price",
    )
    args = parser.parse_args()

    data = json.loads(args.debug_json.read_text(encoding="utf-8"))
    image_path = Path(data["image"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    rows = data.get("debug_rows") or data.get("rows") or []
    target_dir = args.output_dir / image_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for row in rows:
        if not should_export(row, args.include):
            continue
        row_no = str(row.get("No.", "row")).zfill(2)
        side = row.get("Side", "side")
        base_name = f"{row_no}_{side}"

        name_path = ""
        price_path = ""
        if row.get("Name_Cell_Box"):
            name_crop = crop_box(image, row["Name_Cell_Box"])
            name_path = str(target_dir / f"{base_name}_name.png")
            cv2.imwrite(name_path, name_crop)
        if row.get("Price_Cell_Box"):
            price_crop = crop_box(image, row["Price_Cell_Box"])
            price_path = str(target_dir / f"{base_name}_price.png")
            cv2.imwrite(price_path, price_crop)

        manifest_rows.append(
            {
                "No.": row.get("No.", ""),
                "Side": side,
                "Vegetable": row.get("Vegetable", ""),
                "Price": row.get("Price", ""),
                "Raw_Price_OCR": row.get("Raw_Price_OCR", ""),
                "Digit_Model_Price": row.get("Digit_Model_Price", ""),
                "Digit_Model_Confidence": row.get("Digit_Model_Confidence", ""),
                "Name_Crop": name_path,
                "Price_Crop": price_path,
                "Correct_Vegetable": "",
                "Correct_Price": "",
            }
        )

    manifest_path = target_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["No."])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Exported {len(manifest_rows)} review rows -> {target_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
