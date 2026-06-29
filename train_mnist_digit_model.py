import argparse
import gzip
import struct
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from handwritten_digits import DIGIT_SIZE, preprocess_digit
from train_digit_model import load_labeled_samples


MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX magic {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count)


def mnist_to_feature(image: np.ndarray) -> np.ndarray | None:
    canvas = 255 - image
    canvas = cv2.resize(canvas, (DIGIT_SIZE, DIGIT_SIZE), interpolation=cv2.INTER_AREA)
    return preprocess_digit(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local handwritten digit recognizer from MNIST.")
    parser.add_argument("--mnist-dir", type=Path, default=Path("datasets/mnist"))
    parser.add_argument("--output", type=Path, default=Path("models/digit_knn.npz"))
    parser.add_argument("--max-per-digit", type=int, default=1500)
    parser.add_argument("--real-samples-dir", type=Path, default=Path("digit_samples"))
    args = parser.parse_args()

    image_path = args.mnist_dir / "train-images-idx3-ubyte.gz"
    label_path = args.mnist_dir / "train-labels-idx1-ubyte.gz"
    download(MNIST_URLS["train_images"], image_path)
    download(MNIST_URLS["train_labels"], label_path)

    images = read_idx_images(image_path)
    labels = read_idx_labels(label_path)

    samples: list[np.ndarray] = []
    sample_labels: list[int] = []
    counts = {digit: 0 for digit in range(10)}
    for image, label in zip(images, labels):
        label = int(label)
        if counts[label] >= args.max_per_digit:
            continue
        feature = mnist_to_feature(image)
        if feature is None:
            continue
        samples.append(feature.reshape(-1))
        sample_labels.append(label)
        counts[label] += 1
        if all(count >= args.max_per_digit for count in counts.values()):
            break

    real_samples, real_labels = load_labeled_samples(args.real_samples_dir)
    samples.extend(real_samples)
    sample_labels.extend(real_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        samples=np.asarray(samples, dtype=np.float32),
        labels=np.asarray(sample_labels, dtype=np.int64),
        digit_size=np.asarray([DIGIT_SIZE], dtype=np.int64),
    )
    print(f"Saved {len(samples)} samples -> {args.output}")
    print(f"MNIST samples per digit: {counts}")
    if real_samples:
        print(f"Included {len(real_samples)} real labelled samples from {args.real_samples_dir}")


if __name__ == "__main__":
    main()
