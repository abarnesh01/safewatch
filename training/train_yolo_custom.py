"""
SafeWatch — YOLOv8 Training
Fine-tunes YOLOv8 for specialized person detection in surveillance scenarios.
"""

from pathlib import Path
from loguru import logger
import yaml

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def prepare_yolo_data_config(data_dir: str = "data"):
    """Create the YAML config file for YOLOv8 training."""
    data_path = Path(data_dir)
    config = {
        'path': str(data_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'person'
        }
    }
    
    config_file = data_path / "custom_person_detect.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_file)


def train_yolo(epochs: int = 50, imgsz: int = 640):
    """
    Fine-tune YOLOv8 model for better person detection in low-light/surveillance.
    """
    if not YOLO_AVAILABLE:
        logger.error("Ultralytics YOLO not installed. Please run: pip install ultralytics")
        return

    logger.info("Preparing YOLOv8 training data configuration...")
    data_config = prepare_yolo_data_config()

    logger.info("Initializing YOLOv8n pretrained model...")
    model = YOLO("yolov8n.pt")
    
    logger.info("Starting fine-tuning...")
    try:
        model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            device='cpu',  # Force CPU as per requirements
            workers=4,
            project='models/yolo_train',
            name='person_detector',
            exist_ok=True
        )
        logger.info("YOLOv8 training complete. Best model saved in models/yolo_train/person_detector/weights/best.pt")
        
        # Export to ONNX for production inference
        logger.info("Exporting YOLOv8 model to ONNX...")
        model.export(format='onnx', dynamic=True)
        
    except Exception as e:
        logger.error(f"YOLOv8 training failed: {e}")

if __name__ == "__main__":
    train_yolo(epochs=5) # Short run for demonstration
