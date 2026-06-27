import argparse
import sys
import logging
from ultralytics import YOLO

# Setup logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train_yolo")

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Training Script for Tabular Layout Detection"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="datasets/subset_table_rec/data.yaml", 
        help="Path to the data.yaml dataset config file (default: datasets/subset_table_rec/data.yaml)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt", 
        help="Pretrained YOLOv8 model weights to start training from (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Image size for training (default: 640)"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16, 
        help="Batch size for training. Use -1 for auto-batch (default: 16)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to train on, e.g. cpu, cuda, or 0 (default: cpu)"
    )
    parser.add_argument(
        "--project", 
        type=str, 
        default="runs/train", 
        help="Directory to save training runs (default: runs/train)"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="tabular_layout", 
        help="Name of the training run (default: tabular_layout)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting YOLOv8 Model Training")
    logger.info(f"Dataset config: {args.data}")
    logger.info(f"Base weights: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Image size: {args.imgsz}")
    logger.info(f"Batch size: {args.batch}")
    logger.info(f"Training device: {args.device}")
    logger.info(f"Output directory: {args.project}/{args.name}")
    logger.info("=" * 60)
    
    try:
        # Load the model (starts from pretrained yolov8n.pt)
        model = YOLO(args.model)
        
        # Start training
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            exist_ok=True
        )
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Best model weights saved to: {args.project}/{args.name}/weights/best.pt")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
