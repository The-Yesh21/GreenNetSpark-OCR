import argparse
import csv
import json
from pathlib import Path

import cv2


def row_value(row: dict, *keys: str, default=""):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def crop_box(image, box: list[float], pad: int = 6):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]


def should_export(row: dict, include: str) -> bool:
    price = row_value(row, "Price", "price")
    vegetable = row_value(row, "Vegetable", "vegetable")
    price_status = row_value(row, "Price_Status", "price_status")
    vegetable_status = row_value(row, "Vegetable_Status", "vegetable_status")
    if include == "all":
        return True
    if include == "pending-price":
        return price == "PENDING_REVIEW"
    if include == "review-price":
        return price == "PENDING_REVIEW" or str(price_status).startswith("REVIEW_PRICE")
    if include == "pending-name":
        return vegetable == "PENDING_REVIEW"
    if include == "review-name":
        return vegetable == "PENDING_REVIEW" or str(vegetable_status).startswith("REVIEW")
    if include == "pending-any":
        return price == "PENDING_REVIEW" or vegetable == "PENDING_REVIEW"
    raise ValueError(f"Unknown include mode: {include}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO split OCR review crops from debug JSON.")
    parser.add_argument("--debug-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("review_crops"))
    parser.add_argument(
        "--include",
        choices=["pending-price", "review-price", "pending-name", "review-name", "pending-any", "all"],
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
    for index, row in enumerate(rows, start=1):
        if not should_export(row, args.include):
            continue
        row_no = str(row_value(row, "No.", "no", default=index)).zfill(2)
        side = row_value(row, "Side", "side", default="side")
        base_name = f"{row_no}_{side}"

        name_path = ""
        price_path = ""
        name_box = row_value(row, "Name_Cell_Box", "name_box", default=[])
        price_box = row_value(row, "Price_Cell_Box", "price_box", default=[])
        if name_box:
            name_crop = crop_box(image, name_box)
            name_path = str(target_dir / f"{base_name}_name.png")
            cv2.imwrite(name_path, name_crop)
        if price_box:
            price_crop = crop_box(image, price_box)
            price_path = str(target_dir / f"{base_name}_price.png")
            cv2.imwrite(price_path, price_crop)

        manifest_rows.append(
            {
                "No.": row_no,
                "Side": side,
                "Vegetable": row_value(row, "Vegetable", "vegetable"),
                "Vegetable_Status": row_value(row, "Vegetable_Status", "vegetable_status"),
                "Raw_Vegetable_OCR": row_value(row, "Raw_Vegetable_OCR", "raw_vegetable"),
                "Price": row_value(row, "Price", "price"),
                "Price_Status": row_value(row, "Price_Status", "price_status"),
                "Raw_Price_OCR": row_value(row, "Raw_Price_OCR", "raw_price"),
                "Digit_Model_Price": row_value(row, "Digit_Model_Price", "digit_model_price"),
                "Digit_Model_Confidence": row_value(row, "Digit_Model_Confidence", "digit_model_confidence"),
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
