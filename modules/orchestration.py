import csv
import logging
import cv2
import numpy as np
from modules.detection import YoloDetector
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

    def pair_detections(self, detections: list, img_w: float, max_y_diff: float = 100.0) -> tuple:
        """
        Groups detections by left/right side based on the X-axis midpoint of the image,
        and pairs vegetables with prices using a Strict Grid-Anchor Parser.
        
        :param detections: All detections from YOLO.
        :param img_w: Width of the input image.
        :param max_y_diff: Maximum absolute Y distance to allow pairing (for backward compatibility).
        :return: Tuple of lists: (left_pairs, right_pairs)
        """
        # X-Midpoint Enforcement: Calculate and print the calculated midpoint of the image
        midpoint_x = img_w / 2.0
        logger.info(f"X-Midpoint Enforcement: Calculated image midpoint X = {midpoint_x:.1f}")
        
        # 1. Median Row Height: Calculate the median height of all detected boxes to determine row_threshold
        if detections:
            heights = [d['box'][3] - d['box'][1] for d in detections]
            row_threshold = float(np.median(heights))
            logger.info(f"Strict Grid-Anchor Parser: Calculated median height = {row_threshold:.2f}px (row_threshold)")
        else:
            row_threshold = 50.0  # Fallback
            logger.info(f"Strict Grid-Anchor Parser: No detections. Using fallback row_threshold = {row_threshold:.2f}px")
            
        # 2. Bucket Creation: Iterate through all detected boxes and sort them into buckets where Y-center is within ±0.5 * row_threshold
        sorted_boxes = sorted(detections, key=lambda d: self._get_center_y(d['box']))
        row_buckets = []  # list of lists of detections
        
        for box in sorted_boxes:
            y_center = self._get_center_y(box['box'])
            matched_bucket = None
            for bucket in row_buckets:
                avg_y = sum(self._get_center_y(b['box']) for b in bucket) / len(bucket)
                if abs(y_center - avg_y) <= (0.5 * row_threshold):
                    matched_bucket = bucket
                    break
            if matched_bucket is not None:
                matched_bucket.append(box)
            else:
                row_buckets.append([box])
                
        logger.info(f"Strict Grid-Anchor Parser: Grouped into {len(row_buckets)} total Row Buckets using threshold={0.5 * row_threshold:.2f}px")

        # 3. Grid Splitting: For every bucket, sort by X and assign slots 0, 1 (Left) and 2, 3 (Right)
        def pair_row_bucket_slots(bucket):
            left_side_boxes = [d for d in bucket if self._get_center_x(d['box']) < midpoint_x]
            right_side_boxes = [d for d in bucket if self._get_center_x(d['box']) >= midpoint_x]
            
            # Sort each side by X-coordinate
            left_side_boxes.sort(key=lambda d: self._get_center_x(d['box']))
            right_side_boxes.sort(key=lambda d: self._get_center_x(d['box']))
            
            def assign_slots(side_boxes, is_left_side):
                if len(side_boxes) >= 2:
                    # Slot 0/2 is Vegetable, Slot 1/3 is Price
                    return side_boxes[0], side_boxes[1]
                elif len(side_boxes) == 1:
                    det = side_boxes[0]
                    box = det['box']
                    center_x = self._get_center_x(box)
                    width = box[2] - box[0]
                    label_lower = det['label'].lower()
                    
                    is_price = 'price' in label_lower
                    if not is_price:
                        # Position/width heuristic to guess cell type
                        if is_left_side:
                            is_price = center_x > (midpoint_x * 0.65) or width < (midpoint_x * 0.4)
                        else:
                            is_price = center_x > (midpoint_x + (img_w - midpoint_x) * 0.65) or width < ((img_w - midpoint_x) * 0.4)
                    
                    if is_price:
                        return None, det
                    else:
                        return det, None
                return None, None

            left_veg, left_price = assign_slots(left_side_boxes, is_left_side=True)
            right_veg, right_price = assign_slots(right_side_boxes, is_left_side=False)
            
            return left_veg, left_price, right_veg, right_price

        left_pairs = []
        right_pairs = []
        for idx, bucket in enumerate(row_buckets):
            avg_y = sum(self._get_center_y(b['box']) for b in bucket) / len(bucket)
            left_veg, left_price, right_veg, right_price = pair_row_bucket_slots(bucket)
            
            if left_veg or left_price:
                left_pairs.append((left_veg, left_price, idx + 1, avg_y))
            if right_veg or right_price:
                right_pairs.append((right_veg, right_price, idx + 1, avg_y))
                
        return left_pairs, right_pairs

    def process_image(self, 
                      image_path: str, 
                      output_csv_path: str, 
                      conf_threshold: float = 0.25,
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
        detections = self.detector.detect(image_path, conf_threshold=conf_threshold)
        
        # Save visualization if requested
        if visualize_path:
            draw_detections(image_path, detections, visualize_path)
            
        # Read the image once to get dimensions and prepare for cropping
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {image_path}")
        img_h, img_w = image.shape[:2]
            
        # 2. Pairing: split based on image midpoint and pair within halves
        left_pairs, right_pairs = self.pair_detections(detections, img_w=img_w, max_y_diff=max_y_diff)
        
        # Row Normalization: merge and sort the left and right pairs by their Y-center coordinate
        # to process them in top-to-bottom order across the entire page (row-by-row structure)
        combined_pairs = []
        for veg_det, price_det, bucket_idx, avg_y in left_pairs:
            combined_pairs.append(('Left-Grid', veg_det, price_det, bucket_idx, avg_y))
        for veg_det, price_det, bucket_idx, avg_y in right_pairs:
            combined_pairs.append(('Right-Grid', veg_det, price_det, bucket_idx, avg_y))
            
        # Sort combined pairs by the average Y coordinate of the row bucket
        combined_pairs.sort(key=lambda item: item[4])
        
        rows_data = []
        low_confidence_reads = []
        serial_no = 1
        
        logger.info(f"Row Normalization: Processing {len(combined_pairs)} combined rows in top-to-bottom order...")
        
        for side_name, veg_det, price_det, bucket_idx, avg_y in combined_pairs:
            # Vegetable recognition
            if veg_det:
                veg_crop = self._crop_box(image, veg_det['box'])
                raw_veg_text, veg_conf = self.veg_recognizer.recognize(veg_crop)
                # Correct Kannada name
                cleaned_veg = correct_vegetable_name(raw_veg_text)
                
                # Check if vegetable OCR confidence is low (< 0.5)
                if veg_conf < 0.5:
                    low_confidence_reads.append({
                        'box': veg_det['box'],
                        'class': veg_det['label'],
                        'raw_text': raw_veg_text,
                        'confidence': veg_conf
                    })
            else:
                # If vegetable box is missing, mark as MISSING
                cleaned_veg = "MISSING"
                raw_veg_text = "No vegetable box paired"
                veg_conf = 1.0
            
            # Price extraction
            price_conf = 1.0
            raw_ocr_price = ""
            raw_price_text = ""
            if price_det:
                price_crop = self._crop_box(image, price_det['box'])
                cleaned_price, raw_price_text, price_conf = self.price_recognizer.recognize_and_extract(price_crop)
                
                # Check if digits were found. If cleaned_price is empty or "MISSING_PRICE", price defaults to "MISSING_PRICE".
                if cleaned_price == "MISSING_PRICE" or not cleaned_price:
                    cleaned_price = "MISSING_PRICE"
                
                # Store the exact, uncleaned string from the OCR engine (use [Empty OCR] if empty)
                raw_ocr_price = raw_price_text if raw_price_text else "[Empty OCR]"
                
                # Check if price OCR confidence is low (< 0.5)
                if price_conf < 0.5:
                    low_confidence_reads.append({
                        'box': price_det['box'],
                        'class': price_det['label'],
                        'raw_text': raw_price_text,
                        'confidence': price_conf
                    })
            else:
                # If a row has a vegetable name but no price box detected, explicitly mark the price column as 'MISSING'
                cleaned_price = "MISSING"
                raw_ocr_price = "No price box paired"
                raw_price_text = "No price box paired"
                
            rows_data.append({
                'No.': serial_no,
                'Vegetable': cleaned_veg,
                'Price': cleaned_price,
                'raw_ocr_price': raw_ocr_price,
                'Row_Bucket_ID': f"{side_name}-Bucket-{bucket_idx}",
                'Raw_Y_Coord': round(avg_y, 1),
                'raw_veg_diagnostic': raw_veg_text,
                'raw_price_diagnostic': raw_price_text
            })
            
            logger.info(f"Row {serial_no} ({side_name}): Raw='{raw_veg_text}' (conf={veg_conf:.2f}) -> Cleaned='{cleaned_veg}' | Price={cleaned_price} | raw_ocr_price='{raw_ocr_price}' | Row_Bucket_ID={side_name}-Bucket-{bucket_idx} | Raw_Y_Coord={avg_y:.1f}")
            serial_no += 1
            
        # 3. Write to CSV enforcing the schema: [No., Vegetable, Price, raw_ocr_price, Row_Bucket_ID, Raw_Y_Coord]
        # Using extrasaction='ignore' to omit the raw diagnostic trace columns from the written file
        with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csv_file:
            fieldnames = ['No.', 'Vegetable', 'Price', 'raw_ocr_price', 'Row_Bucket_ID', 'Raw_Y_Coord']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
            
            writer.writeheader()
            for row in rows_data:
                writer.writerow(row)
                
        logger.info(f"Successfully wrote {len(rows_data)} rows to {output_csv_path}")
        return rows_data, low_confidence_reads
