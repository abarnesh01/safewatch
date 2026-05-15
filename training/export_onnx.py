"""
SafeWatch — ONNX Export
Exports the trained PyTorch model to ONNX format for optimized CPU inference.
"""

import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from training.train_classifier import ActionLSTM

def export_to_onnx(model_path: str = "models/action_model_best.pt", output_path: str = "models/action_classifier.onnx"):
    """
    Load a trained PyTorch model and export it to ONNX.
    """
    model_file = Path(model_path)
    output_file = Path(output_path)
    
    if not model_file.exists():
        logger.error(f"PyTorch model not found at {model_path}")
        return

    # Configuration (Must match training params)
    input_size = 99      # 33 landmarks * 3 (x, y, vis)
    hidden_size = 256
    num_layers = 2
    num_classes = 9     # normal, fight, fall, etc.
    seq_length = 30     # Fixed sequence length

    logger.info(f"Loading PyTorch model from {model_file}...")
    model = ActionLSTM(input_size, hidden_size, num_layers, num_classes)
    
    try:
        model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        return

    # Create dummy input for tracing
    dummy_input = torch.randn(1, seq_length, input_size)

    logger.info(f"Exporting model to ONNX at {output_file}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_file),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info("ONNX export complete.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")

    # Optional: Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(str(output_file))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verified successfully.")
    except ImportError:
        logger.warning("ONNX library not installed, skipping verification.")
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")

if __name__ == "__main__":
    export_to_onnx()
