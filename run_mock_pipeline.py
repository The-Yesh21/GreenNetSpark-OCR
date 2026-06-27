import logging
import sys
from modules.orchestration import OcrOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mock_pipeline")

def get_mock_detections():
    """
    Returns mock detections for datasets/test1.jpg (1600x1200).
    Defines 5 rows on both left and right sides.
    """
    detections = []
    
    # Define row vertical boundaries (Y coordinates)
    rows_y = [
        (250, 320),   # Row 1
        (350, 420),   # Row 2
        (450, 520),   # Row 3
        (550, 620),   # Row 4
        (650, 720),   # Row 5
        (750, 820),   # Row 6
        (850, 920),   # Row 7
        (950, 1020),  # Row 8
        (1050, 1120), # Row 9
        (1150, 1220)  # Row 10
    ]
    
    # X boundaries
    left_veg_x = (80, 350)
    left_price_x = (380, 520)
    right_veg_x = (620, 900)
    right_price_x = (930, 1100)
    
    for i, (y_min, y_max) in enumerate(rows_y):
        # Left side
        detections.append({
            'box': [left_veg_x[0], y_min, left_veg_x[1], y_max],
            'label': 'left_veg',
            'confidence': 0.95
        })
        detections.append({
            'box': [left_price_x[0], y_min + 5, left_price_x[1], y_max - 5],
            'label': 'left_price',
            'confidence': 0.92
        })
        
        # Right side
        detections.append({
            'box': [right_veg_x[0], y_min, right_veg_x[1], y_max],
            'label': 'right_veg',
            'confidence': 0.94
        })
        detections.append({
            'box': [right_price_x[0], y_min + 5, right_price_x[1], y_max - 5],
            'label': 'right_price',
            'confidence': 0.91
        })
        
    return detections

def main():
    image_path = "datasets/test1.jpg"
    output_csv = "mock_output_v3.csv"
    visualize_path = "mock_detections_overlay.jpg"
    
    logger.info("Initializing OCR Pipeline Orchestrator with mock detector...")
    
    # Initialize the orchestrator (use yolov8n.pt which ultralytics will auto-download)
    orchestrator = OcrOrchestrator(model_path="yolov8n.pt", use_gpu=False)
    
    # Override detector.detect to return our mock boxes
    orchestrator.detector.detect = lambda img_path, conf_threshold=0.25: get_mock_detections()
    
    results, low_confidence_reads = orchestrator.process_image(
        image_path=image_path,
        output_csv_path=output_csv,
        conf_threshold=0.25,
        max_y_diff=100.0,
        visualize_path=visualize_path
    )
    
    logger.info("Pipeline test completed successfully!")
    logger.info(f"Generated output CSV: {output_csv}")
    logger.info(f"Generated visualization: {visualize_path}")
    logger.info("Results preview:")
    for row in results[:10]:
        print(f"Row {row['No.']}: Vegetable: {row['Vegetable']} | Price: {row['Price']}")
        
if __name__ == "__main__":
    main()
