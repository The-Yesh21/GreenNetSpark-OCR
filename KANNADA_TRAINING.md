# Kannada OCR Training Data

The Kannada recognizer cannot be trained from raw board images alone. It needs cropped text images paired with the exact Kannada label.

## Export Review Crops

Run OCR for one image:

```powershell
C:\Python312\python.exe -X utf8 run_one_image_to_csv.py --image datasets\test2.jpeg --output one_by_one_review_test2.csv --reset --debug-dir one_by_one_debug_test2
```

Export rows that need Kannada name review:

```powershell
C:\Python312\python.exe export_review_crops.py --debug-json one_by_one_debug_test2\test2_debug.json --output-dir kannada_review_crops --include review-name
```

Fill `Correct_Vegetable` in:

```text
kannada_review_crops/<image>/manifest.csv
```

## Build Recognition Dataset

After labels are filled:

```powershell
C:\Python312\python.exe import_kannada_rec_samples.py --review-dir kannada_review_crops --output-dir train_data\kannada_rec
```

This creates:

```text
train_data/kannada_rec/images/*.png
train_data/kannada_rec/ka_train.txt
```

Each line in `ka_train.txt` is:

```text
relative/image/path<TAB>correct Kannada label
```

This is the minimum artifact needed for proper Kannada OCR recognizer fine-tuning.
