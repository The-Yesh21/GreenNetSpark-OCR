# Kannada Vegetable Price OCR

Local OCR pipeline for extracting vegetable names in Kannada and prices in English digits from the images in `datasets/`.

## Setup

```powershell
python -m pip install -r requirements.txt
```

PaddleOCR downloads its local model files on first use. No API key or hosted AI service is used.

## Run

```powershell
python -X utf8 paddle_extract.py --image datasets/test1.jpg --output output.csv
```

The same entry point is exposed through `main.py`:

```powershell
python -X utf8 main.py --image datasets/test1.jpg --output output.csv
```

To process all numbered test images:

```powershell
python -X utf8 main.py --dataset datasets --output all_outputs.csv
```

The pipeline also loads a local handwritten digit model from `models/digit_knn.npz` when it exists. It does not blindly trust that model: by default it only replaces PaddleOCR prices when the digit model is confident enough and PaddleOCR is weak/missing.

To force the trained digit model when confident:

```powershell
python -X utf8 main.py --image datasets/test1.jpg --output output.csv --prefer-digit-model
```

## YOLO Splitter

The stronger architecture is:

```text
image -> YOLO cell splitter -> Kannada OCR for veg cells
                         -> digit recognizer for price cells
```

Build an initial YOLO dataset from the green table grid:

```powershell
python build_yolo_cell_dataset.py --source datasets --output datasets/yolo_cell_split
```

Train the splitter:

```powershell
python train_yolo.py --data datasets/yolo_cell_split/data.yaml --model yolov8n.pt --epochs 50 --imgsz 1280 --batch 4 --device cpu
```

Run the YOLO-split pipeline after training:

```powershell
python -X utf8 yolo_split_extract.py --image datasets/test1.jpg --model runs/train/yolo_cell_split/weights/best.pt --output yolo_output.csv --fallback-grid
```

`--fallback-grid` is important while the YOLO model is still young. The extractor checks whether YOLO found a balanced 15-row, 4-column table. If YOLO misses a class or returns too many duplicate cells, it falls back to the deterministic green-grid splitter.

The generated labels are a starting point. For best accuracy, inspect and correct the YOLO labels in a labelling tool before final training. You can create overlays to review them:

```powershell
python visualize_yolo_labels.py --image datasets/test1.jpg --labels datasets/yolo_cell_split/train/labels/test1.txt --output yolo_label_overlay_test1.jpg
python audit_yolo_cell_dataset.py --labels datasets/yolo_cell_split
```

Useful debug files are written by default:

- `ocr_debug.json`: raw Kannada/English OCR reads and the final paired rows
- `ocr_debug_overlay.jpg`: green boxes for vegetable candidates and purple boxes for price candidates
- `yolo_split_debug.json`: YOLO/grid cells, final rows, and crop boxes for review exports

## Training

The immediate extraction path uses PaddleOCR's local pretrained OCR models. The existing `train_yolo.py` script can still train a YOLO layout detector from a labelled table-structure dataset, but it is optional and does not replace PaddleOCR.

For handwritten English prices, train the local digit model:

```powershell
python train_digit_model.py --output models/digit_knn.npz --samples-per-digit 900
```

For stronger handwritten-number pretraining, use MNIST:

```powershell
python train_mnist_digit_model.py --mnist-dir datasets/mnist --output models/digit_knn.npz --max-per-digit 1500
```

To improve it with real handwriting from your images, first run OCR to create `ocr_debug.json`, then bootstrap confident price crops:

```powershell
python -X utf8 main.py --image datasets/test1.jpg --output output.csv
python bootstrap_digit_samples.py --image datasets/test1.jpg --debug-json ocr_debug.json --output-dir digit_samples
python train_mnist_digit_model.py --mnist-dir datasets/mnist --output models/digit_knn.npz --real-samples-dir digit_samples
```

For the YOLO/grid split path, export the rows where prices are still missing:

```powershell
python -X utf8 yolo_split_extract.py --image datasets/test1.jpg --model runs/train/yolo_cell_split/weights/best.pt --output yolo_output.csv --debug-json yolo_split_debug.json --fallback-grid
python export_review_crops.py --debug-json yolo_split_debug.json --output-dir review_crops --include pending-price
```

Fill `Correct_Price` in the generated `review_crops/<image>/manifest.csv`. Those crops are the best source for improving the handwritten price recognizer because they come from the real board handwriting that PaddleOCR missed.

The best version will come from manually checking `digit_samples/0` through `digit_samples/9` and moving any wrong crops into the correct digit folder before retraining. True OCR fine-tuning needs labelled text crops/transcriptions; the current `test1.jpg`, `test2.jpeg`, etc. images are not enough by themselves to fine-tune a Kannada OCR recognizer.
