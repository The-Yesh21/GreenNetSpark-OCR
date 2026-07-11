import argparse
import csv
from pathlib import Path

import cv2

from handwritten_digits import segment_digits


def resolve_crop_path(raw_path: str, manifest_path: Path) -> Path:
    crop_path = Path(raw_path)
    if crop_path.exists():
        return crop_path
    candidate = manifest_path.parent / raw_path
    if candidate.exists():
        return candidate
    return crop_path


def clean_price(value: str) -> str:
    return "".join(ch for ch in str(value).strip() if ch.isdigit())


def import_manifest(manifest_path: Path, output_dir: Path, overwrite: bool) -> tuple[int, int, list[str]]:
    saved = 0
    skipped = 0
    warnings: list[str] = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        price = clean_price(row.get("Correct_Price", ""))
        crop_value = row.get("Price_Crop", "")
        if not price or not crop_value:
            skipped += 1
            continue

        crop_path = resolve_crop_path(crop_value, manifest_path)
        crop = cv2.imread(str(crop_path))
        if crop is None:
            skipped += 1
            warnings.append(f"Missing crop: {crop_value}")
            continue

        # Apply 2x upscaling and 10px white padding to match recognition pipeline preprocessing
        crop = cv2.resize(crop, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        segments = segment_digits(crop)
        if len(segments) != len(price):
            skipped += 1
            warnings.append(
                f"Skipped {crop_path}: corrected price {price!r} has {len(price)} digits, "
                f"but segmentation found {len(segments)}"
            )
            continue

        row_no = str(row.get("No.", "row")).zfill(2)
        side = row.get("Side", "side")
        manifest_stem = manifest_path.parent.name
        for digit_index, ((_, box), digit) in enumerate(zip(segments, price), start=1):
            sx1, sy1, sx2, sy2 = box
            digit_crop = crop[sy1:sy2, sx1:sx2]
            digit_dir = output_dir / digit
            digit_dir.mkdir(parents=True, exist_ok=True)
            out_path = digit_dir / f"{manifest_stem}_{row_no}_{side}_{digit_index}_{digit}.png"
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            cv2.imwrite(str(out_path), digit_crop)
            saved += 1

    return saved, skipped, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Import corrected review price crops into digit_samples.")
    parser.add_argument("--manifest", type=Path, action="append", help="One manifest.csv to import. Can be repeated.")
    parser.add_argument("--review-dir", type=Path, default=Path("review_crops"), help="Import every manifest.csv below this directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("digit_samples"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for digit in range(10):
        (args.output_dir / str(digit)).mkdir(parents=True, exist_ok=True)

    manifests = args.manifest or sorted(args.review_dir.glob("*/manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"No manifests found. Pass --manifest or create manifests under {args.review_dir}")

    total_saved = 0
    total_skipped = 0
    all_warnings: list[str] = []
    for manifest in manifests:
        saved, skipped, warnings = import_manifest(manifest, args.output_dir, args.overwrite)
        total_saved += saved
        total_skipped += skipped
        all_warnings.extend(warnings)
        print(f"{manifest}: saved {saved}, skipped {skipped}")

    if all_warnings:
        print("Warnings:")
        for warning in all_warnings:
            print(f"- {warning}")
    print(f"Total saved {total_saved} digit crops into {args.output_dir}; skipped {total_skipped}")


if __name__ == "__main__":
    main()
