import argparse
import csv
import json
import tempfile
from pathlib import Path

import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO

from handwritten_digits import DigitKnnRecognizer
from paddle_extract import DEFAULT_DIGIT_MODEL_PATH, _correct_name, _digits, _load_vegetable_dictionary
from yolo_cells import CellBox, YOLO_CELL_CLASSES, detect_grid_cell_boxes


def make_ocr(lang: str) -> PaddleOCR:
    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def crop_box(image, box: list[float], pad: int = 4):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image[y1:y2, x1:x2]


def read_crop_ocr(ocr: PaddleOCR, crop) -> tuple[str, float]:
    if crop.size == 0:
        return "", 0.0
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = Path(handle.name)
    cv2.imwrite(str(path), crop)
    try:
        result = ocr.predict(str(path))
    finally:
        path.unlink(missing_ok=True)
    if not result:
        return "", 0.0
    page = result[0]
    texts = [str(text).strip() for text in page.get("rec_texts", []) if str(text).strip()]
    scores = [float(score) for score in page.get("rec_scores", [])]
    if not texts:
        return "", 0.0
    return " ".join(texts), (sum(scores) / len(scores) if scores else 0.0)


def detect_cells(model_path: Path, image_path: Path, confidence: float, imgsz: int) -> list[CellBox]:
    model = YOLO(str(model_path))
    result = model(str(image_path), conf=confidence, iou=0.45, imgsz=imgsz)[0]
    boxes: list[CellBox] = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, str(cls_id))
        if label not in YOLO_CELL_CLASSES:
            continue
        boxes.append(CellBox(label=label, box=[float(v) for v in box.xyxy[0].tolist()], confidence=float(box.conf[0])))
    return sorted(boxes, key=lambda item: (item.cy, item.cx))


def pair_cells(cells: list[CellBox], max_y_gap: float) -> list[tuple[CellBox, CellBox | None]]:
    rows: list[tuple[CellBox, CellBox | None]] = []
    for side in ("left", "right"):
        names = sorted([cell for cell in cells if cell.label == f"{side}_veg"], key=lambda item: item.cy)
        prices = sorted([cell for cell in cells if cell.label == f"{side}_price"], key=lambda item: item.cy)
        unused_prices = prices[:]
        for name in names:
            candidates = [price for price in unused_prices if abs(price.cy - name.cy) <= max_y_gap]
            price = min(candidates, key=lambda item: abs(item.cy - name.cy), default=None)
            if price:
                unused_prices.remove(price)
            rows.append((name, price))
    return sorted(rows, key=lambda pair: (pair[0].label.split("_")[0], pair[0].cy))


def write_csv(path: Path, rows: list[dict]) -> None:
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
        "Yolo_Name_Confidence",
        "Yolo_Price_Confidence",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO splitter -> Kannada OCR and handwritten digit OCR.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("runs/train/yolo_cell_split/weights/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("yolo_split_output.csv"))
    parser.add_argument("--debug-json", type=Path, default=Path("yolo_split_debug.json"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-y-gap", type=float, default=40.0)
    parser.add_argument("--digit-model", type=Path, default=DEFAULT_DIGIT_MODEL_PATH)
    parser.add_argument("--digit-model-min-confidence", type=float, default=0.58)
    parser.add_argument("--prefer-digit-model", action="store_true")
    parser.add_argument("--fallback-grid", action="store_true", help="Use grid-derived boxes if YOLO detects too few cells")
    parser.add_argument("--min-yolo-cells", type=int, default=20, help="Minimum YOLO detections before accepting YOLO output")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"YOLO model not found: {args.model}. Train it with train_yolo.py first.")

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)

    kannada_ocr = make_ocr("ka")
    english_ocr = make_ocr("en")
    digit_recognizer = DigitKnnRecognizer(args.digit_model) if args.digit_model.exists() else None
    dictionary = _load_vegetable_dictionary()
    cells = detect_cells(args.model, args.image, args.conf, args.imgsz)
    detector_source = "yolo"
    if args.fallback_grid and len(cells) < args.min_yolo_cells:
        cells = detect_grid_cell_boxes(args.image)
        detector_source = "grid_fallback"

    rows: list[dict] = []
    for idx, (name_cell, price_cell) in enumerate(pair_cells(cells, args.max_y_gap), start=1):
        side = name_cell.label.split("_")[0]
        raw_name, name_score = read_crop_ocr(kannada_ocr, crop_box(image, name_cell.box))
        veg_name = _correct_name(raw_name, dictionary) if raw_name else "PENDING_REVIEW"

        raw_price = ""
        paddle_price = ""
        price_score = 0.0
        digit_price = ""
        digit_conf = 0.0
        if price_cell:
            price_crop = crop_box(image, price_cell.box)
            raw_price, price_score = read_crop_ocr(english_ocr, price_crop)
            paddle_price = _digits(raw_price)
            if digit_recognizer:
                prediction = digit_recognizer.recognize(price_crop)
                digit_price = prediction.text
                digit_conf = prediction.confidence

        use_digit = (
            digit_price
            and digit_conf >= args.digit_model_min_confidence
            and (args.prefer_digit_model or not paddle_price or price_score < 0.85)
        )
        final_price = digit_price if use_digit else paddle_price

        rows.append({
            "No.": idx,
            "Side": side,
            "Vegetable": veg_name,
            "Price": final_price or "PENDING_REVIEW",
            "Vegetable_Confidence": round(name_score, 4),
            "Price_Confidence": round(price_score, 4),
            "Raw_Price_OCR": raw_price,
            "Digit_Model_Price": digit_price,
            "Digit_Model_Confidence": round(digit_conf, 4),
            "Price_Source": "digit_model" if use_digit else ("paddleocr" if paddle_price else "pending_review"),
            "Yolo_Name_Confidence": round(name_cell.confidence, 4),
            "Yolo_Price_Confidence": round(price_cell.confidence, 4) if price_cell else 0.0,
        })

    write_csv(args.output, rows)
    args.debug_json.write_text(
        json.dumps(
            {"image": str(args.image), "detector_source": detector_source, "cells": [cell.__dict__ for cell in cells], "rows": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Detected {len(cells)} cells via {detector_source}; extracted {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
