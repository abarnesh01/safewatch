# SafeWatch Training Guide — Google Colab

## Overview
These scripts are designed to run on **Google Colab** with a GPU runtime.
DO NOT run these on your local machine unless you have a dedicated GPU.

## Training Workflow

### Step 1: Dataset Preparation
```python
# Run in Google Colab
!pip install mediapipe opencv-python ultralytics scikit-learn

from dataset_prep import DatasetPrep
prep = DatasetPrep()
prep.download_all()
prep.extract_frames()
prep.extract_poses()
prep.split_dataset()
prep.generate_manifest()
```

### Step 2: Train Action Classifier
```python
from train_classifier import ActionClassifierTrainer
trainer = ActionClassifierTrainer()
trainer.train(epochs=50)
trainer.evaluate()
trainer.plot_curves()
```

### Step 3: Export to ONNX
```python
from export_onnx import ONNXExporter
exporter = ONNXExporter()
exporter.export("checkpoints/best_model.pt", "models/action_classifier.onnx")
exporter.validate()
exporter.benchmark()
```

### Step 4: Download Models
Download the exported `.onnx` file and place it in your local
`safewatch/models/` directory.

## Datasets Used
1. **RWF-2000** — Fight/non-fight video dataset (2000 clips)
2. **UCF-Crime** — 13 anomaly classes from surveillance footage
3. **Le2i Fall Detection** — Fall detection video dataset
4. **Hockey Fight** — Hockey fight/non-fight clips

## Hardware Requirements
- Google Colab with T4 GPU (free tier works)
- At least 15GB disk space for datasets
- Training takes ~2-4 hours on T4

## Estimated Model Performance
- Action Classifier: ~85% accuracy on validation set
- Per-class F1: varies by class (fight ~0.88, fall ~0.82, normal ~0.92)
