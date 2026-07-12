from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DIGIT_SIZE = 28


@dataclass
class DigitPrediction:
    text: str
    confidence: float
    boxes: list[list[int]]


def preprocess_digit(gray_or_bgr: np.ndarray) -> np.ndarray | None:
    if gray_or_bgr.size == 0:
        return None
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    if w < 3 or h < 6:
        return None

    digit = binary[y:y + h, x:x + w]
    side = max(w, h) + 10
    canvas = np.zeros((side, side), dtype=np.uint8)
    xoff = (side - w) // 2
    yoff = (side - h) // 2
    canvas[yoff:yoff + h, xoff:xoff + w] = digit
    resized = cv2.resize(canvas, (DIGIT_SIZE, DIGIT_SIZE), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0).reshape(1, -1)


def segment_digits(crop: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    
    h, w = gray.shape[:2]
    has_white_border = False
    if h > 20 and w > 20:
        top_mean = np.mean(gray[:10, :])
        bottom_mean = np.mean(gray[-10:, :])
        left_mean = np.mean(gray[:, :10])
        right_mean = np.mean(gray[:, -10:])
        if top_mean > 248 and bottom_mean > 248 and left_mean > 248 and right_mean > 248:
            has_white_border = True
            
    if has_white_border:
        inner_gray = gray[10:-10, 10:-10]
    else:
        inner_gray = gray
        
    inner_gray = cv2.GaussianBlur(inner_gray, (3, 3), 0)
    _, binary_inner = cv2.threshold(inner_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if has_white_border:
        binary = cv2.copyMakeBorder(binary_inner, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    else:
        binary = binary_inner
        
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts: list[tuple[int, int, int, int]] = []
    h_img, w_img = binary.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if h < max(8, h_img * 0.22):
            continue
        if w < 3 or area < 8:
            continue
        if w > w_img * 0.95 and h > h_img * 0.95:
            continue
        parts.append((x, y, w, h))

    if not parts:
        return []

    parts.sort(key=lambda item: item[0])
    merged: list[tuple[int, int, int, int]] = []
    for x, y, w, h in parts:
        if not merged:
            merged.append((x, y, w, h))
            continue
        px, py, pw, ph = merged[-1]
        gap = x - (px + pw)
        overlaps_y = min(y + h, py + ph) - max(y, py)
        if gap <= 2 and overlaps_y > 0:
            nx = min(px, x)
            ny = min(py, y)
            nx2 = max(px + pw, x + w)
            ny2 = max(py + ph, y + h)
            merged[-1] = (nx, ny, nx2 - nx, ny2 - ny)
        else:
            merged.append((x, y, w, h))

    digits: list[tuple[np.ndarray, list[int]]] = []
    for x, y, w, h in merged:
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        sub_boxes = _split_wide_component(binary[y1:y2, x1:x2], x1, y1)
        for sx1, sy1, sx2, sy2 in sub_boxes:
            feature = preprocess_digit(crop[sy1:sy2, sx1:sx2])
            if feature is not None:
                digits.append((feature, [sx1, sy1, sx2, sy2]))
    return digits


def _split_wide_component(binary_crop: np.ndarray, x_offset: int, y_offset: int) -> list[list[int]]:
    h, w = binary_crop.shape[:2]
    if w <= h * 0.9:
        return [[x_offset, y_offset, x_offset + w, y_offset + h]]

    projection = np.sum(binary_crop > 0, axis=0).astype(np.float32)
    if projection.max(initial=0) <= 0:
        return [[x_offset, y_offset, x_offset + w, y_offset + h]]

    smooth = np.convolve(projection, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    threshold = max(1.0, smooth.max() * 0.18)
    candidate_cols = np.where(smooth <= threshold)[0]
    valid = candidate_cols[(candidate_cols > w * 0.18) & (candidate_cols < w * 0.82)]
    if valid.size == 0:
        estimated_digits = int(round(w / max(1.0, h * 0.62)))
        if estimated_digits <= 1:
            return [[x_offset, y_offset, x_offset + w, y_offset + h]]
        cuts = [int(w * i / estimated_digits) for i in range(1, estimated_digits)]
    else:
        groups: list[list[int]] = []
        current = [int(valid[0])]
        for col in valid[1:]:
            col = int(col)
            if col - current[-1] <= 2:
                current.append(col)
            else:
                groups.append(current)
                current = [col]
        groups.append(current)
        cuts = [int(np.mean(group)) for group in groups]

    boundaries = [0] + cuts + [w]
    boxes: list[list[int]] = []
    for left, right in zip(boundaries, boundaries[1:]):
        if right - left < 4:
            continue
        part = binary_crop[:, left:right]
        coords = cv2.findNonZero(part)
        if coords is None:
            continue
        px, py, pw, ph = cv2.boundingRect(coords)
        boxes.append([x_offset + left + px, y_offset + py, x_offset + left + px + pw, y_offset + py + ph])

    return boxes or [[x_offset, y_offset, x_offset + w, y_offset + h]]


class DigitKnnRecognizer:
    def __init__(self, model_path: Path, k: int = 5):
        payload = np.load(model_path)
        self.samples = payload["samples"].astype(np.float32)
        self.labels = payload["labels"].astype(np.int64)
        self.k = k

    def predict_feature(self, feature: np.ndarray) -> tuple[str, float]:
        diff = self.samples - feature.astype(np.float32)
        distances = np.einsum("ij,ij->i", diff, diff)
        nearest = np.argpartition(distances, self.k)[:self.k]
        votes = self.labels[nearest]
        counts = np.bincount(votes, minlength=10)
        digit = int(np.argmax(counts))
        vote_conf = float(counts[digit]) / float(self.k)
        mean_dist = float(np.mean(distances[nearest]))
        distance_conf = 1.0 / (1.0 + mean_dist)
        return str(digit), max(0.0, min(1.0, (vote_conf + distance_conf) / 2.0))

    def recognize(self, crop: np.ndarray) -> DigitPrediction:
        digits = segment_digits(crop)
        if not digits:
            return DigitPrediction("", 0.0, [])

        chars: list[str] = []
        confidences: list[float] = []
        boxes: list[list[int]] = []
        for feature, box in digits:
            digit, confidence = self.predict_feature(feature)
            chars.append(digit)
            confidences.append(confidence)
            boxes.append(box)

        return DigitPrediction("".join(chars), float(np.mean(confidences)), boxes)
