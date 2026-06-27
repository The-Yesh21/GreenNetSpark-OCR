import logging
import cv2
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YoloDetector:
    """
    Detection Module utilizing YOLOv8 to identify layout structures:
    'left_veg', 'left_price', 'right_veg', and 'right_price'.
    """
    def __init__(self, model_path: str):
        """
        Initializes the YOLOv8 model.
        :param model_path: Path to the trained YOLOv8 model weights (.pt file).
        """
        logger.info(f"Initializing YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        
        # Expected classes
        self.expected_classes = {'left_veg', 'left_price', 'right_veg', 'right_price'}
        
    def detect(self, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.6) -> list:
        """
        Runs inference on the input image and returns sorted detections.
        
        :param image_path: Path to the input image.
        :param conf_threshold: Confidence threshold for YOLO detections (default: 0.25).
        :param iou_threshold: IoU threshold for NMS to prevent suppressing faint adjacent boxes (default: 0.6).
        :return: List of dictionaries containing coordinates, class label, and confidence.
                 Sorted by Y-coordinate (top to bottom) to maintain row integrity.
        """
        logger.info(f"Running YOLOv8 inference on {image_path} with conf={conf_threshold}, iou={iou_threshold}...")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")
            
        img_h, img_w = img.shape[:2]
        
        # Run inference with specified confidence and IoU thresholds (ensuring high resolution imgsz=1280 for margin/top-row detection)
        results = self.model(img, conf=conf_threshold, iou=iou_threshold, imgsz=1280)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates (x_min, y_min, x_max, y_max)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Get the class name from model names
                label = self.model.names.get(cls_id, str(cls_id))
                
                # Context-Aware Cropping: add a padding factor of 15% (broadened to 20% for price boxes)
                w = x2 - x1
                h = y2 - y1
                is_price = 'price' in label.lower()
                pad_factor = 0.20 if is_price else 0.15
                pad_x = w * pad_factor
                pad_y = h * pad_factor
                
                # Apply padding and clamp to image boundaries
                x1_pad = max(0.0, x1 - pad_x)
                y1_pad = max(0.0, y1 - pad_y)
                x2_pad = min(float(img_w), x2 + pad_x)
                y2_pad = min(float(img_h), y2 + pad_y)
                
                detections.append({
                    'box': [x1_pad, y1_pad, x2_pad, y2_pad],
                    'label': label,
                    'confidence': conf
                })
                
        # Count and log the detections for expected classes
        class_counts = {cls: 0 for cls in self.expected_classes}
        for d in detections:
            lbl = d['label']
            if lbl in class_counts:
                class_counts[lbl] += 1
            else:
                class_counts[lbl] = class_counts.get(lbl, 0) + 1
                
        counts_str = ", ".join(f"'{k}': {v}" for k, v in class_counts.items())
        logger.info(f"Detection Diagnostic Counts -> {counts_str}")
        
        # Sort by Y-coordinate of the center of the bounding box
        detections.sort(key=lambda d: (d['box'][1] + d['box'][3]) / 2)
        
        logger.info(f"Detected {len(detections)} objects.")
        for d in detections:
            logger.debug(f"Class: {d['label']}, Conf: {d['confidence']:.2f}, Box: {d['box']}")
            
        return detections
