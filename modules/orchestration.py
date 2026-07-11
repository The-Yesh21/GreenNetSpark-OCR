import csv
import logging
import cv2
import numpy as np
from modules.detection import YoloDetector, split_image_into_zones
from modules.recognition import VegetableRecognizer, PriceRecognizer
from utils.name_correction import correct_vegetable_name
from utils.image_utils import draw_detections

logger = logging.getLogger(__name__)

class OcrOrchestrator:
    """
    Orchestration & Cleaning Module.
    Coordinates the YOLOv8 detector, VegetableRecognizer, and PriceRecognizer.
    Pairs vegetable and price detections by Y-axis proximity,
    cleans the data, and writes the output to a CSV file.
    """
    def __init__(self, model_path: str, use_gpu: bool = False):
        """
        Initializes the orchestrator by loading modules.
        
        :param model_path: Path to the trained YOLOv8 model weights (.pt).
        :param use_gpu: Whether to use GPU for PaddleOCR.
        """
        self.detector = YoloDetector(model_path)
        self.veg_recognizer = VegetableRecognizer(use_gpu=use_gpu)
        self.price_recognizer = PriceRecognizer(use_gpu=use_gpu)

    def _get_center_y(self, box: list) -> float:
        """Returns the Y-coordinate of the center of a bounding box."""
        return (box[1] + box[3]) / 2.0

    def _crop_box(self, image: np.ndarray, box: list, padding: int = 5) -> np.ndarray:
        """
        Crops the bounding box from the image with safety padding.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        # Add padding safely within image dimensions
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return image[y1:y2, x1:x2]

    def _get_center_x(self, box: list) -> float:
        """Returns the X-coordinate of the center of a bounding box."""
        return (box[0] + box[2]) / 2.0

    def process_image(self, 
                      image_path: str, 
                      output_csv_path: str, 
                      conf_threshold: float = 0.15,
                      max_y_diff: float = 100.0,
                      visualize_path: str = None) -> tuple:
        """
        Runs the full pipeline on a single image.
        
        :param image_path: Path to the input image.
        :param output_csv_path: Path where the CSV results will be saved.
        :param conf_threshold: YOLO detection confidence threshold.
        :param max_y_diff: Max Y distance allowed for pairing.
        :param visualize_path: Path to save detection visualization image.
        :return: Tuple of (rows_data, low_confidence_reads)
        """
        # 1. Detection
        detections = self.detector.detect(image_path, conf_threshold=conf_threshold, iou_threshold=0.45)
        
        # Confidence-Based Filtering: reject any detection with confidence score below 0.35
        detections = [d for d in detections if d.get('confidence', 0.0) >= 0.35]
        
        # Grid Fallback: if YOLO cell coverage is incomplete (< 20), fallback to grid layout splitting
        if len(detections) < 20:
            from yolo_cells import detect_grid_cell_boxes
            logger.info("YOLO detections sparse (< 20). Falling back to grid layout splitting.")
            grid_cells = detect_grid_cell_boxes(image_path)
            detections = []
            for cell in grid_cells:
                detections.append({
                    'box': cell.box,
                    'label': cell.label,
                    'confidence': cell.confidence
                })
        
        # Save visualization if requested
        if visualize_path:
            draw_detections(image_path, detections, visualize_path)
            
        # Read the image once to get dimensions and prepare for cropping
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {image_path}")
        img_h, img_w = image.shape[:2]
            
        # 2. Four-Zone Slicing / Spatial Splitting
        zones = split_image_into_zones(detections, img_w=img_w)
        
        # Validation: Print the count of items in each zone to the console before pairing
        zone_counts_msg = (
            f"Validation - Zone Counts:\n"
            f"  Left-Veg: {len(zones['Left-Veg'])}\n"
            f"  Left-Price: {len(zones['Left-Price'])}\n"
            f"  Right-Veg: {len(zones['Right-Veg'])}\n"
            f"  Right-Price: {len(zones['Right-Price'])}"
        )
        print(zone_counts_msg)
        logger.info(zone_counts_msg)
        
        low_confidence_reads = []
        
        # 3. Zone-Specific Recognition
        # Route Left-Veg and Right-Veg strictly through name_correction dictionary
        for zone_name in ['Left-Veg', 'Right-Veg']:
            for d in zones[zone_name]:
                crop = self._crop_box(image, d['box'])
                # Preprocess: Upscale 2x and add 10px white padding to improve OCR
                if crop.size > 0:
                    crop = cv2.resize(crop, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                raw_veg_text, veg_conf = self.veg_recognizer.recognize(crop)
                cleaned_veg = correct_vegetable_name(raw_veg_text)
                
                d['ocr_text'] = cleaned_veg
                d['raw_text'] = raw_veg_text
                d['confidence'] = veg_conf
                
                if veg_conf < 0.5:
                    low_confidence_reads.append({
                        'box': d['box'],
                        'class': d['label'],
                        'raw_text': raw_veg_text,
                        'confidence': veg_conf
                    })
                    
        # Route Left-Price and Right-Price zones strictly through the digit-only parser
        for zone_name in ['Left-Price', 'Right-Price']:
            for d in zones[zone_name]:
                crop = self._crop_box(image, d['box'])
                # Preprocess: Upscale 2x and add 10px white padding to improve OCR
                if crop.size > 0:
                    crop = cv2.resize(crop, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                cleaned_price, raw_price_text, price_conf = self.price_recognizer.extract_price_strict(crop)
                
                if cleaned_price in ("MISSING_PRICE", "0") or not cleaned_price:
                    cleaned_price = "MISSING_PRICE"
                
                raw_ocr_price = raw_price_text if raw_price_text else "[Empty OCR]"
                
                d['ocr_text'] = cleaned_price
                d['raw_text'] = raw_price_text
                d['confidence'] = price_conf
                d['raw_ocr_price'] = raw_ocr_price
                
                if price_conf < 0.5:
                    low_confidence_reads.append({
                        'box': d['box'],
                        'class': d['label'],
                        'raw_text': raw_price_text,
                        'confidence': price_conf
                    })
                    
        # Sort boxes within each zone by their Y-coordinate (using center Y for stability)
        for zone_name in zones:
            zones[zone_name].sort(key=lambda d: self._get_center_y(d['box']))
            
        left_veg = [d['ocr_text'] for d in zones['Left-Veg']]
        left_price = [d['ocr_text'] for d in zones['Left-Price']]
        right_veg = [d['ocr_text'] for d in zones['Right-Veg']]
        right_price = [d['ocr_text'] for d in zones['Right-Price']]
        
        # Ensure the lists have the same length by padding them to allow simultaneous iteration
        max_len = max(len(left_veg), len(left_price), len(right_veg), len(right_price))
        while len(left_veg) < max_len:
            left_veg.append("MISSING")
        while len(left_price) < max_len:
            left_price.append("MISSING")
        while len(right_veg) < max_len:
            right_veg.append("MISSING")
        while len(right_price) < max_len:
            right_price.append("MISSING")
            
        # Result Sanitization: If a 'Veg' cell is found but corresponding 'Price' is missing/invalid, use 'PENDING_REVIEW'
        for i in range(max_len):
            if left_veg[i] != "MISSING" and left_price[i] in ("MISSING_PRICE", "MISSING"):
                left_price[i] = "PENDING_REVIEW"
            if right_veg[i] != "MISSING" and right_price[i] in ("MISSING_PRICE", "MISSING"):
                right_price[i] = "PENDING_REVIEW"
                
        # Count empty price cells (missing, invalid, or pending review)
        empty_price_count = 0
        for i in range(max_len):
            if left_price[i] in ("PENDING_REVIEW", "MISSING_PRICE", "MISSING"):
                empty_price_count += 1
            if right_price[i] in ("PENDING_REVIEW", "MISSING_PRICE", "MISSING"):
                empty_price_count += 1
            
        rows_data = []
        
        # 4. Forced Data Injection & Print Validation
        for i in range(max_len):
            # Print Validation statement
            print(f"Row {i}: Veg={left_veg[i]}, Price={left_price[i]}")
            
            row = {
                'No.': i,
                'Left_Veg': left_veg[i],
                'Left_Price': left_price[i],
                'Right_Veg': right_veg[i],
                'Right_Price': right_price[i],
                # Add compatibility fields for downstream processors/loggers
                'Vegetable': left_veg[i],
                'Price': left_price[i],
                'raw_veg_diagnostic': left_veg[i],
                'raw_price_diagnostic': left_price[i]
            }
            rows_data.append(row)
            
        # Missing Row Diagnostic: log a 'MISSING_BOX' alert if it detects fewer than 28 pairs
        if max_len < 28:
            row_y_centers = []
            for i in range(max_len):
                y_vals = []
                for zone_name in ['Left-Veg', 'Left-Price', 'Right-Veg', 'Right-Price']:
                    if i < len(zones[zone_name]):
                        d = zones[zone_name][i]
                        y_vals.append(self._get_center_y(d['box']))
                if y_vals:
                    row_y_centers.append(sum(y_vals) / len(y_vals))
            row_y_centers.sort()
            
            # Calculate average Y vertical spacing between rows
            spacings = [row_y_centers[j+1] - row_y_centers[j] for j in range(len(row_y_centers) - 1)]
            avg_spacing = sum(spacings) / len(spacings) if spacings else 100.0
            
            # Estimate Y center of missing top row
            expected_top_y = row_y_centers[0] - avg_spacing if row_y_centers else 100.0
            
            # Get dimensions of vegetable boxes to calculate bounds
            veg_boxes = []
            for zone_name in ['Left-Veg', 'Right-Veg']:
                for d in zones[zone_name]:
                    veg_boxes.append(d['box'])
            if veg_boxes:
                avg_w = sum(b[2] - b[0] for b in veg_boxes) / len(veg_boxes)
                avg_h = sum(b[3] - b[1] for b in veg_boxes) / len(veg_boxes)
            else:
                avg_w = 0.20 * img_w
                avg_h = 0.05 * img_h
                
            left_veg_boxes = [d['box'] for d in zones['Left-Veg']]
            avg_lv_x = sum((b[0] + b[2]) / 2.0 for b in left_veg_boxes) / len(left_veg_boxes) if left_veg_boxes else 0.15 * img_w
            
            right_veg_boxes = [d['box'] for d in zones['Right-Veg']]
            avg_rv_x = sum((b[0] + b[2]) / 2.0 for b in right_veg_boxes) / len(right_veg_boxes) if right_veg_boxes else 0.65 * img_w
            
            lv_coords = [avg_lv_x - avg_w / 2.0, expected_top_y - avg_h / 2.0, avg_lv_x + avg_w / 2.0, expected_top_y + avg_h / 2.0]
            rv_coords = [avg_rv_x - avg_w / 2.0, expected_top_y - avg_h / 2.0, avg_rv_x + avg_w / 2.0, expected_top_y + avg_h / 2.0]
            
            lv_coords = [max(0.0, lv_coords[0]), max(0.0, lv_coords[1]), min(float(img_w), lv_coords[2]), min(float(img_h), lv_coords[3])]
            rv_coords = [max(0.0, rv_coords[0]), max(0.0, rv_coords[1]), min(float(img_w), rv_coords[2]), min(float(img_h), rv_coords[3])]
            
            alert_msg = (
                f"MISSING_BOX Alert: Detected {max_len} pairs, which is fewer than the expected 28.\n"
                f"Based on average grid spacing of {avg_spacing:.1f}px, the missing top-row vegetable boxes are expected at:\n"
                f"  Left-Veg Expected Box: [{', '.join(f'{c:.1f}' for c in lv_coords)}]\n"
                f"  Right-Veg Expected Box: [{', '.join(f'{c:.1f}' for c in rv_coords)}]"
            )
            logger.warning(alert_msg)
            print(alert_msg)
            
        # Validation Print: add a summary print: 'Pipeline completed. Rows mapped: X. Empty price cells: Y.'
        summary_msg = f"Pipeline completed. Rows mapped: {max_len}. Empty price cells: {empty_price_count}."
        print(summary_msg)
        logger.info(summary_msg)

        # Raw Save: Write this list of dictionaries directly to output_csv_path
        try:
            with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csv_file:
                fieldnames = ['No.', 'Left_Veg', 'Left_Price', 'Right_Veg', 'Right_Price']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
                
                writer.writeheader()
                for row in rows_data:
                    writer.writerow(row)
            logger.info(f"Successfully wrote {len(rows_data)} rows to {output_csv_path}")
        except PermissionError:
            err_msg = f"ERROR: Permission denied writing to '{output_csv_path}'. Please close the file in Excel and re-run."
            logger.error(err_msg)
            print(err_msg)
            raise PermissionError(f"Permission denied: '{output_csv_path}'. Please close the file in Excel.")
            
        return rows_data, low_confidence_reads
