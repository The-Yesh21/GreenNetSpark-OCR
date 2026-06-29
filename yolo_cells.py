from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


YOLO_CELL_CLASSES = ["left_veg", "left_price", "right_veg", "right_price"]


@dataclass
class CellBox:
    label: str
    box: list[float]
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0


def _cluster_positions(values: np.ndarray, min_gap: int = 8) -> list[int]:
    if values.size == 0:
        return []
    values = np.asarray(sorted(set(int(v) for v in values)))
    clusters: list[list[int]] = [[int(values[0])]]
    for value in values[1:]:
        value = int(value)
        if value - clusters[-1][-1] <= min_gap:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [int(round(float(np.mean(cluster)))) for cluster in clusters]


def _green_grid_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 25, 25], dtype=np.uint8)
    upper = np.array([100, 255, 235], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.medianBlur(mask, 3)


def _line_positions(mask: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = mask.shape[:2]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(18, h // 28)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(18, w // 24), 1))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)

    x_projection = np.sum(vertical > 0, axis=0)
    y_projection = np.sum(horizontal > 0, axis=1)
    x_candidates = np.where(x_projection > max(8, h * 0.10))[0]
    y_candidates = np.where(y_projection > max(8, w * 0.12))[0]
    return _cluster_positions(x_candidates), _cluster_positions(y_candidates)


def _fallback_lines(image: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = image.shape[:2]
    left = int(w * 0.12)
    right = int(w * 0.86)
    top = int(h * 0.20)
    bottom = int(h * 0.88)
    xs = [
        left,
        int(left + (right - left) * 0.30),
        int(left + (right - left) * 0.50),
        int(left + (right - left) * 0.72),
        right,
    ]
    rows = 16
    ys = [int(top + (bottom - top) * i / rows) for i in range(rows + 1)]
    return xs, ys


def detect_grid_cell_boxes(image_path: Path, min_rows: int = 8, expected_rows: int | None = 15) -> list[CellBox]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    mask = _green_grid_mask(image)
    xs, ys = _line_positions(mask)
    if len(xs) < 5 or len(ys) < min_rows + 1:
        xs, ys = _fallback_lines(image)

    xs = sorted(xs)
    ys = sorted(ys)
    if len(xs) > 5:
        xs = [xs[0], xs[1], xs[len(xs) // 2], xs[-2], xs[-1]]
    if len(xs) >= 5:
        widths = [max(1, xs[i + 1] - xs[i]) for i in range(4)]
        if max(widths) / min(widths) > 3.0:
            xs, fallback_ys = _fallback_lines(image)
            if len(ys) < min_rows + 1:
                ys = fallback_ys

    row_intervals: list[tuple[int, int]] = []
    for top, bottom in zip(ys[:-1], ys[1:]):
        row_h = bottom - top
        if row_h < 14:
            continue
        row_center = (top + bottom) / 2.0
        if row_center < image.shape[0] * 0.20:
            continue
        if row_center > image.shape[0] * 0.84:
            continue
        row_intervals.append((top, bottom))

    if row_intervals:
        heights = np.asarray([bottom - top for top, bottom in row_intervals], dtype=float)
        median_h = float(np.median(heights))
        if median_h > 0:
            filtered = [
                interval for interval, height in zip(row_intervals, heights)
                if height <= median_h * 2.5
            ]
            if len(filtered) >= min_rows:
                row_intervals = filtered
    if expected_rows is not None and len(row_intervals) != expected_rows and row_intervals:
        top = row_intervals[0][0]
        bottom = row_intervals[-1][1]
        uniform = np.linspace(top, bottom, expected_rows + 1)
        row_intervals = [(int(round(uniform[i])), int(round(uniform[i + 1]))) for i in range(expected_rows)]

    boxes: list[CellBox] = []
    labels = YOLO_CELL_CLASSES
    pad_x = 4.0
    pad_y = 3.0
    for top, bottom in row_intervals:
        for col_idx, label in enumerate(labels):
            if col_idx + 1 >= len(xs):
                continue
            x1 = float(xs[col_idx] + pad_x)
            x2 = float(xs[col_idx + 1] - pad_x)
            y1 = float(top + pad_y)
            y2 = float(bottom - pad_y)
            if x2 - x1 < 10 or y2 - y1 < 8:
                continue
            boxes.append(CellBox(label=label, box=[x1, y1, x2, y2]))
    return boxes


def write_yolo_label(label_path: Path, image_shape: tuple[int, int], boxes: list[CellBox]) -> None:
    h, w = image_shape[:2]
    class_to_id = {name: idx for idx, name in enumerate(YOLO_CELL_CLASSES)}
    lines: list[str] = []
    for item in boxes:
        x1, y1, x2, y2 = item.box
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{class_to_id[item.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
