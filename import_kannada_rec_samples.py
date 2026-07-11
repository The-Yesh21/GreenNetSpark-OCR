import argparse
import csv
import shutil
from pathlib import Path


def resolve_crop_path(raw_path: str, manifest_path: Path) -> Path:
    crop_path = Path(raw_path)
    if crop_path.exists():
        return crop_path
    candidate = manifest_path.parent / raw_path
    if candidate.exists():
        return candidate
    return crop_path


def clean_label(value: str) -> str:
    return " ".join(str(value).strip().split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Import corrected Kannada name crops for OCR recognition training.")
    parser.add_argument("--manifest", type=Path, action="append", help="One review manifest.csv. Can be repeated.")
    parser.add_argument("--review-dir", type=Path, default=Path("kannada_review_crops"))
    parser.add_argument("--output-dir", type=Path, default=Path("train_data/kannada_rec"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifests = args.manifest or sorted(args.review_dir.glob("*/manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"No manifests found. Pass --manifest or create manifests under {args.review_dir}")

    image_dir = args.output_dir / "images"
    label_path = args.output_dir / "ka_train.txt"
    image_dir.mkdir(parents=True, exist_ok=True)

    label_rows: list[str] = []
    saved = 0
    skipped = 0
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            label = clean_label(row.get("Correct_Vegetable", ""))
            crop_value = row.get("Name_Crop", "")
            if not label or not crop_value:
                skipped += 1
                continue
            crop_path = resolve_crop_path(crop_value, manifest)
            if not crop_path.exists():
                skipped += 1
                continue
            stem = manifest.parent.name
            row_no = str(row.get("No.", "row")).zfill(2)
            side = row.get("Side", "side")
            target_name = f"{stem}_{row_no}_{side}_name{crop_path.suffix.lower() or '.png'}"
            target_path = image_dir / target_name
            if not target_path.exists() or args.overwrite:
                shutil.copy2(crop_path, target_path)
            label_rows.append(f"images/{target_name}\t{label}")
            saved += 1

    label_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
    print(f"Saved {saved} Kannada OCR samples -> {args.output_dir}")
    print(f"Skipped {skipped} rows without Correct_Vegetable or Name_Crop")
    print(f"Label file: {label_path}")


if __name__ == "__main__":
    main()
