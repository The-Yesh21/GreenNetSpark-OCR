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
            
        rows_data = []
        
        # 4. Forced Data Injection & Print Validation
        for i in range(max_len):
            # Print Validation statement: Inside this new loop, add a print(f'Row {i}: Veg={left_veg[i]}, Price={left_price[i]}') statement.
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
            
        # Raw Save: Write this list of dictionaries directly to output_csv_path
        with open(output_csv_path, mode='w', encoding='utf-8-sig', newline='') as csv_file:
            fieldnames = ['No.', 'Left_Veg', 'Left_Price', 'Right_Veg', 'Right_Price']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
            
            writer.writeheader()
            for row in rows_data:
                writer.writerow(row)
                
        logger.info(f"Successfully wrote {len(rows_data)} rows to {output_csv_path}")
        return rows_data, low_confidence_reads
