import argparse
import csv
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from paddleocr import PaddleOCR

from handwritten_digits import DigitKnnRecognizer


KANNADA_RE = re.compile(r"[\u0C80-\u0CFF]")
PRICE_RE = re.compile(r"\d+")

ROOT = Path(__file__).resolve().parent
VEG_DICT_PATH = ROOT / "veg_dictionary.json"
DEFAULT_DIGIT_MODEL_PATH = ROOT / "models" / "digit_knn.npz"


@dataclass
class TextBox:
    text: str
    score: float
    box: list[float]

    @property
    def x1(self) -> float:
        return self.box[0]

    @property
    def y1(self) -> float:
        return self.box[1]

    @property
    def x2(self) -> float:
        return self.box[2]

    @property
    def y2(self) -> float:
        return self.box[3]

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)


def _make_ocr(lang: str) -> PaddleOCR:
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _normalize_box(box) -> list[float]:
    arr = np.asarray(box, dtype=float)
    if arr.ndim == 1 and arr.size == 4:
        x1, y1, x2, y2 = arr.tolist()
        return [x1, y1, x2, y2]

    arr = arr.reshape(-1, 2)
    xs = arr[:, 0]
    ys = arr[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _read_ocr(ocr: PaddleOCR, image_path: str) -> list[TextBox]:
    result = ocr.predict(image_path)
    if not result:
        return []

    page = result[0]
    texts = page.get("rec_texts", [])
    scores = page.get("rec_scores", [])
    boxes = page.get("rec_boxes")
    if boxes is None:
        boxes = page.get("rec_polys", [])

    reads: list[TextBox] = []
    for text, score, box in zip(texts, scores, boxes):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        reads.append(TextBox(cleaned, float(score), _normalize_box(box)))
    return reads


def _digits(text: str) -> str:
    translated = text.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}))
    matches = PRICE_RE.findall(translated)
    return "".join(matches)


def _load_vegetable_dictionary() -> dict[str, str]:
    if not VEG_DICT_PATH.exists():
        return {}
    return json.loads(VEG_DICT_PATH.read_text(encoding="utf-8"))


def _correct_name(text: str, dictionary: dict[str, str]) -> str:
    cleaned = re.sub(r"\s+", " ", text.replace("--", "-")).strip()
    if not cleaned:
        return cleaned
    if cleaned in dictionary:
        return dictionary[cleaned]

    keys = list(dictionary.keys())
    matches = difflib.get_close_matches(cleaned, keys, n=1, cutoff=0.56)
    if matches:
        return dictionary[matches[0]]
    return cleaned


def _is_header_or_footer(box: TextBox, image_h: int) -> bool:
    text = box.text.lower()
    if box.cy < image_h * 0.16:
        return True
    if box.cy > image_h * 0.91:
        return True
    return any(token in text for token in ("mys", "apmc", "mar", "2026", "taal", "date"))


def _looks_like_vegetable(box: TextBox, image_h: int, min_score: float) -> bool:
    if box.score < min_score:
        return False
    if _is_header_or_footer(box, image_h):
        return False
    kannada_chars = len(KANNADA_RE.findall(box.text))
    return kannada_chars >= 2


def _looks_like_price(box: TextBox, image_h: int, min_score: float) -> bool:
    if box.score < min_score:
        return False
    if _is_header_or_footer(box, image_h):
        return False
    return bool(_digits(box.text))


def _side(box: TextBox, image_w: int) -> str:
    return "left" if box.cx < image_w / 2.0 else "right"


def _crop_box(image: np.ndarray, box: list[float], pad: int = 4) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]


def _infer_price_box(name: TextBox, side: str, image_w: int, image_h: int) -> list[float]:
    row_h = max(42.0, name.height * 2.1)
    y1 = max(0.0, name.cy - row_h / 2.0)
    y2 = min(float(image_h), name.cy + row_h / 2.0)
    if side == "left":
        x1 = min(float(image_w), max(name.x2 + 10.0, image_w * 0.34))
        x2 = min(float(image_w), image_w * 0.50)
    else:
        x1 = min(float(image_w), max(name.x2 + 10.0, image_w * 0.70))
        x2 = min(float(image_w), image_w * 0.86)
    return [x1, y1, x2, y2]


def _pair_side(
    names: list[TextBox],
    prices: list[TextBox],
    max_y_gap: float,
    image: np.ndarray,
    side: str,
    digit_recognizer: DigitKnnRecognizer | None,
    digit_model_min_confidence: float,
    prefer_digit_model: bool,
) -> list[dict]:
    rows: list[dict] = []
    unused_prices = prices[:]
    image_h, image_w = image.shape[:2]

    for name in sorted(names, key=lambda item: item.cy):
        candidates = [
            price for price in unused_prices
            if price.x1 >= name.x2 + 8.0 and abs(price.cy - name.cy) <= max(max_y_gap, name.height * 1.25)
        ]
        price = min(candidates, key=lambda item: abs(item.cy - name.cy), default=None)
        if price is not None:
            unused_prices.remove(price)

        paddle_price = _digits(price.text) if price else ""
        digit_price = ""
        digit_confidence = 0.0
        digit_box = price.box if price else _infer_price_box(name, side, image_w, image_h)

        if digit_recognizer is not None:
            prediction = digit_recognizer.recognize(_crop_box(image, digit_box))
            digit_price = prediction.text
            digit_confidence = prediction.confidence

        use_digit_price = (
            digit_price
            and digit_confidence >= digit_model_min_confidence
            and (prefer_digit_model or not paddle_price or (price is not None and price.score < 0.85))
        )
        final_price = digit_price if use_digit_price else paddle_price

        rows.append({
            "vegetable": name.text,
            "price": final_price or "PENDING_REVIEW",
            "vegetable_confidence": round(name.score, 4),
            "price_confidence": round(price.score, 4) if price else 0.0,
            "raw_price": price.text if price else "",
            "digit_model_price": digit_price,
            "digit_model_confidence": round(digit_confidence, 4),
            "price_source": "digit_model" if use_digit_price else ("paddleocr" if paddle_price else "pending_review"),
            "name_box": [round(v, 2) for v in name.box],
            "price_box": [round(v, 2) for v in digit_box] if digit_box else [],
        })
    return rows


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    fieldnames = [
        "No.",
        "Side",
        "Vegetable",
        "Price",
        "Vegetable_Confidence",
        "Price_Confidence",
        "Raw_Price_OCR",
        "Digit_Model_Price",
        "Digit_Model_Confidence",
        "Price_Source",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow({
                "No.": idx,
                "Side": row["side"],
                "Vegetable": row["vegetable"],
                "Price": row["price"],
                "Vegetable_Confidence": row["vegetable_confidence"],
                "Price_Confidence": row["price_confidence"],
                "Raw_Price_OCR": row["raw_price"],
                "Digit_Model_Price": row["digit_model_price"],
                "Digit_Model_Confidence": row["digit_model_confidence"],
                "Price_Source": row["price_source"],
            })


def _write_batch_csv(path: Path, rows: Iterable[dict]) -> None:
    fieldnames = [
        "Image",
        "No.",
        "Side",
        "Vegetable",
        "Price",
        "Vegetable_Confidence",
        "Price_Confidence",
        "Raw_Price_OCR",
        "Digit_Model_Price",
        "Digit_Model_Confidence",
        "Price_Source",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow({
                "Image": row["image"],
                "No.": idx,
                "Side": row["side"],
                "Vegetable": row["vegetable"],
                "Price": row["price"],
                "Vegetable_Confidence": row["vegetable_confidence"],
                "Price_Confidence": row["price_confidence"],
                "Raw_Price_OCR": row["raw_price"],
                "Digit_Model_Price": row["digit_model_price"],
                "Digit_Model_Confidence": row["digit_model_confidence"],
                "Price_Source": row["price_source"],
            })


def _draw_debug(image_path: Path, out_path: Path, names: list[TextBox], prices: list[TextBox]) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for box in names:
        x1, y1, x2, y2 = map(int, box.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 160, 0), 2)
    for box in prices:
        x1, y1, x2, y2 = map(int, box.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (180, 0, 180), 2)

    cv2.imwrite(str(out_path), image)


def extract_image(
    image_path: Path,
    output_csv: Path,
    debug_json: Path | None,
    debug_image: Path | None,
    min_name_score: float,
    min_price_score: float,
    max_y_gap: float,
    digit_model_path: Path | None,
    digit_model_min_confidence: float,
    prefer_digit_model: bool,
) -> list[dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_h, image_w = image.shape[:2]

    kannada_ocr = _make_ocr("ka")
    english_ocr = _make_ocr("en")

    kannada_reads = _read_ocr(kannada_ocr, str(image_path))
    english_reads = _read_ocr(english_ocr, str(image_path))
    vegetable_dictionary = _load_vegetable_dictionary()

    names = [box for box in kannada_reads if _looks_like_vegetable(box, image_h, min_name_score)]
    prices = [box for box in english_reads if _looks_like_price(box, image_h, min_price_score)]
    for box in names:
        box.text = _correct_name(box.text, vegetable_dictionary)

    digit_recognizer = None
    if digit_model_path and digit_model_path.exists():
        digit_recognizer = DigitKnnRecognizer(digit_model_path)

    rows: list[dict] = []
    for side in ("left", "right"):
        side_names = [box for box in names if _side(box, image_w) == side]
        side_prices = [box for box in prices if _side(box, image_w) == side]
        for row in _pair_side(
            side_names,
            side_prices,
            max_y_gap,
            image,
            side,
            digit_recognizer,
            digit_model_min_confidence,
            prefer_digit_model,
        ):
            row["side"] = side
            rows.append(row)

    rows.sort(key=lambda row: (row["side"], row["name_box"][1] if row["name_box"] else 0))
    _write_csv(output_csv, rows)

    if debug_json:
        payload = {
            "image": str(image_path),
            "rows": rows,
            "kannada_reads": [box.__dict__ for box in kannada_reads],
            "english_reads": [box.__dict__ for box in english_reads],
        }
        debug_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if debug_image:
        _draw_debug(image_path, debug_image, names, prices)

    return rows


def iter_dataset_images(dataset_dir: Path) -> list[Path]:
    patterns = ("test*.jpg", "test*.jpeg", "test*.png")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(dataset_dir.glob(pattern))
    return sorted(images, key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Kannada vegetable names and English prices using local PaddleOCR.")
    parser.add_argument("--image", type=Path, help="Input image path, for example datasets/test1.jpg")
    parser.add_argument("--dataset", type=Path, help="Process every test*.jpg/jpeg/png image in this folder")
    parser.add_argument("--output", default=Path("output.csv"), type=Path, help="CSV output path")
    parser.add_argument("--debug-json", default=Path("ocr_debug.json"), type=Path, help="Raw OCR/debug JSON path")
    parser.add_argument("--debug-image", default=Path("ocr_debug_overlay.jpg"), type=Path, help="Debug overlay image path")
    parser.add_argument("--min-name-score", default=0.15, type=float, help="Minimum Kannada OCR confidence for vegetable candidates")
    parser.add_argument("--min-price-score", default=0.20, type=float, help="Minimum English OCR confidence for price candidates")
    parser.add_argument("--max-y-gap", default=45.0, type=float, help="Maximum vertical gap for name-price pairing")
    parser.add_argument("--digit-model", default=DEFAULT_DIGIT_MODEL_PATH, type=Path, help="Local trained digit model path")
    parser.add_argument("--digit-model-min-confidence", default=0.58, type=float, help="Minimum digit model confidence before using it")
    parser.add_argument("--prefer-digit-model", action="store_true", help="Use the trained digit model over PaddleOCR when it is confident")
    args = parser.parse_args()
    if not args.image and not args.dataset:
        parser.error("Provide --image for one image or --dataset for a folder.")

    if args.dataset:
        all_rows: list[dict] = []
        for image_path in iter_dataset_images(args.dataset):
            image_rows = extract_image(
                image_path=image_path,
                output_csv=args.output.with_name(f"{image_path.stem}_output.csv"),
                debug_json=None,
                debug_image=None,
                min_name_score=args.min_name_score,
                min_price_score=args.min_price_score,
                max_y_gap=args.max_y_gap,
                digit_model_path=args.digit_model,
                digit_model_min_confidence=args.digit_model_min_confidence,
                prefer_digit_model=args.prefer_digit_model,
            )
            for row in image_rows:
                row["image"] = image_path.name
            all_rows.extend(image_rows)
            print(f"{image_path.name}: {len(image_rows)} rows")
        _write_batch_csv(args.output, all_rows)
        print(f"Extracted {len(all_rows)} total rows -> {args.output}")
    else:
        rows = extract_image(
            image_path=args.image,
            output_csv=args.output,
            debug_json=args.debug_json,
            debug_image=args.debug_image,
            min_name_score=args.min_name_score,
            min_price_score=args.min_price_score,
            max_y_gap=args.max_y_gap,
            digit_model_path=args.digit_model,
            digit_model_min_confidence=args.digit_model_min_confidence,
            prefer_digit_model=args.prefer_digit_model,
        )
        print(f"Extracted {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
