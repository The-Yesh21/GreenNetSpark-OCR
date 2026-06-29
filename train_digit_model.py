import argparse
from pathlib import Path

import cv2
import numpy as np

from handwritten_digits import DIGIT_SIZE, preprocess_digit


FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
]


def render_digit(digit: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.full((64, 64), 255, dtype=np.uint8)
    font = int(rng.choice(FONTS))
    scale = float(rng.uniform(1.25, 2.35))
    thickness = int(rng.integers(2, 6))
    text = str(digit)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int((64 - tw) / 2 + rng.integers(-7, 8))
    y = int((64 + th) / 2 + rng.integers(-7, 8))
    cv2.putText(canvas, text, (x, y), font, scale, 0, thickness, cv2.LINE_AA)

    angle = float(rng.uniform(-18.0, 18.0))
    matrix = cv2.getRotationMatrix2D((32, 32), angle, float(rng.uniform(0.85, 1.12)))
    matrix[0, 1] += float(rng.uniform(-0.08, 0.08))
    matrix[1, 0] += float(rng.uniform(-0.08, 0.08))
    warped = cv2.warpAffine(canvas, matrix, (64, 64), borderValue=255)

    if rng.random() < 0.35:
        warped = cv2.GaussianBlur(warped, (3, 3), 0)
    noise = rng.normal(0, rng.uniform(0, 10), warped.shape).astype(np.int16)
    noisy = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def load_labeled_samples(samples_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    samples: list[np.ndarray] = []
    labels: list[int] = []
    if not samples_dir.exists():
        return samples, labels

    for digit_dir in samples_dir.iterdir():
        if not digit_dir.is_dir() or not digit_dir.name.isdigit():
            continue
        label = int(digit_dir.name)
        if label < 0 or label > 9:
            continue
        for image_path in list(digit_dir.glob("*.png")) + list(digit_dir.glob("*.jpg")) + list(digit_dir.glob("*.jpeg")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            feature = preprocess_digit(image) if image is not None else None
            if feature is not None:
                samples.append(feature.reshape(-1))
                labels.append(label)
    return samples, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local handwritten English digit recognizer.")
    parser.add_argument("--output", type=Path, default=Path("models/digit_knn.npz"))
    parser.add_argument("--samples-per-digit", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--real-samples-dir",
        type=Path,
        default=Path("digit_samples"),
        help="Optional labelled folders: digit_samples/0/*.png ... digit_samples/9/*.png",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    samples: list[np.ndarray] = []
    labels: list[int] = []

    for digit in range(10):
        for _ in range(args.samples_per_digit):
            image = render_digit(digit, rng)
            feature = preprocess_digit(image)
            if feature is not None:
                samples.append(feature.reshape(-1))
                labels.append(digit)

    real_samples, real_labels = load_labeled_samples(args.real_samples_dir)
    samples.extend(real_samples)
    labels.extend(real_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        samples=np.asarray(samples, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        digit_size=np.asarray([DIGIT_SIZE], dtype=np.int64),
    )
    print(f"Saved {len(samples)} digit samples -> {args.output}")
    if real_samples:
        print(f"Included {len(real_samples)} real labelled samples from {args.real_samples_dir}")


if __name__ == "__main__":
    main()
