import argparse
import csv
from pathlib import Path

from paddle_extract import DEFAULT_DIGIT_MODEL_PATH, extract_image


FIELDNAMES = [
    "Image",
    "No.",
    "Side",
    "Vegetable",
    "Raw_Vegetable_OCR",
    "Vegetable_Dictionary_Match",
    "Vegetable_Dictionary_Confidence",
    "Vegetable_Status",
    "Price",
    "Vegetable_Confidence",
    "Price_Confidence",
    "Raw_Price_OCR",
    "Digit_Model_Price",
    "Digit_Model_Confidence",
    "Price_Source",
]


def append_rows(output_path: Path, image_name: str, rows: list[dict], reset: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = reset or not output_path.exists() or output_path.stat().st_size == 0
    mode = "w" if reset else "a"
    with output_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "Image": image_name,
                    "No.": idx,
                    "Side": row["side"],
                    "Vegetable": row["vegetable"],
                    "Raw_Vegetable_OCR": row.get("raw_vegetable", ""),
                    "Vegetable_Dictionary_Match": row.get("vegetable_dictionary_match", ""),
                    "Vegetable_Dictionary_Confidence": row.get("vegetable_dictionary_confidence", 0.0),
                    "Vegetable_Status": row.get("vegetable_status", ""),
                    "Price": row["price"],
                    "Vegetable_Confidence": row["vegetable_confidence"],
                    "Price_Confidence": row["price_confidence"],
                    "Raw_Price_OCR": row["raw_price"],
                    "Digit_Model_Price": row["digit_model_price"],
                    "Digit_Model_Confidence": row["digit_model_confidence"],
                    "Price_Source": row["price_source"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dataset image and append its OCR rows to a review CSV.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("one_by_one_review.csv"))
    parser.add_argument("--reset", action="store_true", help="Overwrite the review CSV before appending this image.")
    parser.add_argument("--debug-dir", type=Path, default=Path("one_by_one_debug"))
    parser.add_argument("--digit-model", type=Path, default=DEFAULT_DIGIT_MODEL_PATH)
    parser.add_argument("--digit-model-min-confidence", type=float, default=0.58)
    parser.add_argument("--prefer-digit-model", action="store_true")
    args = parser.parse_args()

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    image_output = args.debug_dir / f"{args.image.stem}_output.csv"
    debug_json = args.debug_dir / f"{args.image.stem}_debug.json"
    debug_image = args.debug_dir / f"{args.image.stem}_overlay.jpg"

    rows = extract_image(
        image_path=args.image,
        output_csv=image_output,
        debug_json=debug_json,
        debug_image=debug_image,
        min_name_score=0.15,
        min_price_score=0.20,
        max_y_gap=45.0,
        digit_model_path=args.digit_model,
        digit_model_min_confidence=args.digit_model_min_confidence,
        prefer_digit_model=args.prefer_digit_model,
    )
    append_rows(args.output, args.image.name, rows, args.reset)
    pending_prices = sum(1 for row in rows if row["price"] == "PENDING_REVIEW")
    pending_names = sum(1 for row in rows if row["vegetable"] == "PENDING_REVIEW")
    print(
        f"{args.image.name}: appended {len(rows)} rows to {args.output}; "
        f"pending_prices={pending_prices}; pending_names={pending_names}"
    )
    print(f"Per-image CSV: {image_output}")
    print(f"Debug JSON: {debug_json}")
    print(f"Overlay: {debug_image}")


if __name__ == "__main__":
    main()
