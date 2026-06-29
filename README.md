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

Useful debug files are written by default:

- `ocr_debug.json`: raw Kannada/English OCR reads and the final paired rows
- `ocr_debug_overlay.jpg`: green boxes for vegetable candidates and purple boxes for price candidates

## Training

The immediate extraction path uses PaddleOCR's local pretrained OCR models. The existing `train_yolo.py` script can still train a YOLO layout detector from a labelled table-structure dataset, but it is optional and does not replace PaddleOCR.

For handwritten English prices, train the local digit model:

```powershell
python train_digit_model.py --output models/digit_knn.npz --samples-per-digit 900
```

To improve it with real handwriting from your images, first run OCR to create `ocr_debug.json`, then bootstrap confident price crops:

```powershell
python -X utf8 main.py --image datasets/test1.jpg --output output.csv
python bootstrap_digit_samples.py --image datasets/test1.jpg --debug-json ocr_debug.json --output-dir digit_samples
python train_digit_model.py --output models/digit_knn.npz --real-samples-dir digit_samples
```

The best version will come from manually checking `digit_samples/0` through `digit_samples/9` and moving any wrong crops into the correct digit folder before retraining. True OCR fine-tuning needs labelled text crops/transcriptions; the current `test1.jpg`, `test2.jpeg`, etc. images are not enough by themselves to fine-tune a Kannada OCR recognizer.
