import argparse
import sys
import logging
from modules.orchestration import OcrOrchestrator

# Setup logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(
        description="Modular OCR Pipeline for Tabular Handwritten Data (YOLOv8 + PaddleOCR)"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Path to the input table image"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="best.pt", 
        help="Path to the trained YOLOv8 model weights (.pt file)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.csv", 
        help="Path to save the final CSV file (default: output.csv)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="YOLOv8 confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--max_y_diff", 
        type=float, 
        default=100.0, 
        help="Max Y-axis distance (in pixels) for pairing vegetables and prices (default: 100.0)"
    )
    parser.add_argument(
        "--visualize", 
        type=str, 
        default="detections_overlay.jpg", 
        help="Path to save visual bounding box detections (default: detections_overlay.jpg)"
    )
    parser.add_argument(
        "--gpu", 
        action="store_true", 
        help="Enable GPU execution for PaddleOCR (default: False/CPU)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting Modular OCR Pipeline")
    logger.info(f"Input Image: {args.image}")
    logger.info(f"YOLO Model: {args.model}")
    logger.info(f"Output CSV: {args.output}")
    logger.info(f"Visualization Output: {args.visualize}")
    logger.info(f"Using GPU: {args.gpu}")
    logger.info("=" * 60)
    
    try:
        # Initialize the Orchestrator
        orchestrator = OcrOrchestrator(model_path=args.model, use_gpu=args.gpu)
        
        # Process the image
        results, low_confidence_reads = orchestrator.process_image(
            image_path=args.image,
            output_csv_path=args.output,
            conf_threshold=args.conf,
            max_y_diff=args.max_y_diff,
            visualize_path=args.visualize
        )
        
        # Diagnostic trace printing raw OCR readings
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC TRACE - RAW OCR READS (BEFORE POST-PROCESSING):")
        for row in results:
            logger.info(
                f"Row {row['No.']} | "
                f"Raw Vegetable OCR: '{row.get('raw_veg_diagnostic', '')}' | "
                f"Raw Price OCR: '{row.get('raw_price_diagnostic', '')}'"
            )
        logger.info("=" * 60)
        
        # Generate missing_rows.log file for validation
        log_path = "missing_rows.log"
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("--- LOW CONFIDENCE OCR READS LOG ---\n")
            log_file.write(f"Source Image: {args.image}\n")
            log_file.write(f"Threshold: < 0.50\n")
            log_file.write(f"Total Low Confidence Reads Found: {len(low_confidence_reads)}\n")
            log_file.write("=" * 60 + "\n\n")
            
            for read in low_confidence_reads:
                box_str = ", ".join(f"{v:.1f}" for v in read['box'])
                log_file.write(
                    f"[{read['class']}] Box: [{box_str}] | "
                    f"Raw Text: '{read['raw_text']}' | "
                    f"Confidence: {read['confidence']:.4f}\n"
                )
                
        logger.info("Processing completed successfully!")
        logger.info(f"Extracted {len(results)} rows. CSV saved to: {args.output}")
        logger.info(f"Validation log generated: {log_path} ({len(low_confidence_reads)} low-confidence items)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
