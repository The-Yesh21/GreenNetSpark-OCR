import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def draw_detections(image_path: str, detections: list, output_path: str):
    """
    Draws bounding boxes and labels on the image and saves it.
    
    :param image_path: Path to the original input image.
    :param detections: List of detection dictionaries.
    :param output_path: Path where the output visualization image will be saved.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image for drawing: {image_path}")
        return
        
    # Define colors (BGR) for the bounding boxes
    colors = {
        'left_veg': (255, 100, 0),     # Blue-ish
        'left_price': (0, 255, 100),   # Green-ish
        'right_veg': (0, 100, 255),    # Orange-ish
        'right_price': (200, 0, 200)   # Purple-ish
    }
    
    for det in detections:
        box = det['box']
        label = det['label']
        conf = det['confidence']
        
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(label, (128, 128, 128))
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Put text label
        text = f"{label} ({conf:.2f})"
        font_scale = 0.5
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size for background box
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw small background box for text readability
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # Write text
        cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    cv2.imwrite(output_path, image)
    logger.info(f"Saved visualization overlay to {output_path}")
