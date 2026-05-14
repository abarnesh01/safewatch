# SafeWatch Models

This directory stores pre-trained and custom models:

| Model | File | Description |
|---|---|---|
| YOLOv8n | `yolov8n.pt` | Person detection (auto-downloaded) |
| Action Classifier | `action_classifier.onnx` | LSTM-based threat classification |
| Custom YOLO | `custom_threat_yolo.pt` | Fine-tuned object detection |

## Getting Models

### YOLOv8n (Auto-download)
The YOLOv8n model downloads automatically on first run.

### Action Classifier (Train on Colab)
1. Follow the guide in `training/README_COLAB.md`
2. Train the LSTM model on Google Colab
3. Export to ONNX
4. Place `action_classifier.onnx` in this directory

### Custom YOLO (Optional)
Fine-tune YOLOv8 on custom data using `training/train_yolo_custom.py`.

## Fallback Mode
SafeWatch runs in **rule-based fallback mode** when ONNX models are not present.
Detection still works using skeleton geometry and velocity heuristics.
