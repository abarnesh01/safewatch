"""
SafeWatch — Train Custom YOLOv8
Fine-tune YOLOv8 on custom threat detection data.
Designed for Google Colab.
"""

from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class CustomYOLOTrainer:
    """Fine-tune YOLOv8 on custom threat detection dataset."""

    def __init__(
        self,
        base_model: str = "yolov8n.pt",
        data_yaml: str = "data/custom_yolo/data.yaml",
        output_dir: str = "runs/train",
    ):
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics not available. Install with: pip install ultralytics")

        self._base_model = base_model
        self._data_yaml = data_yaml
        self._output_dir = output_dir

    def __repr__(self) -> str:
        return f"CustomYOLOTrainer(base='{self._base_model}')"

    def prepare_dataset_yaml(
        self,
        train_dir: str = "data/custom_yolo/train",
        val_dir: str = "data/custom_yolo/val",
        classes: list[str] = None,
        output_path: str = "data/custom_yolo/data.yaml",
    ):
        """
        Create the YOLO dataset YAML config.

        Args:
            train_dir: Training images directory
            val_dir: Validation images directory
            classes: List of class names
            output_path: Path to write YAML config
        """
        if classes is None:
            classes = ["person", "fight", "weapon", "fallen_person"]

        import yaml

        data_config = {
            "train": str(Path(train_dir).resolve()),
            "val": str(Path(val_dir).resolve()),
            "nc": len(classes),
            "names": classes,
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"✅ Dataset YAML saved to: {out}")
        self._data_yaml = str(out)

    def train(
        self,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 20,
    ):
        """
        Train the custom YOLO model.

        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            patience: Early stopping patience
        """
        model = YOLO(self._base_model)

        results = model.train(
            data=self._data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=self._output_dir,
            name="safewatch_yolo",
            save=True,
            plots=True,
            verbose=True,
        )

        print(f"\n✅ Training complete!")
        print(f"Best model: {results.save_dir / 'weights' / 'best.pt'}")

        return results

    def export_model(
        self,
        model_path: str = None,
        output_path: str = "models/custom_threat_yolo.pt",
    ):
        """
        Copy the best trained model to the models directory.

        Args:
            model_path: Path to best.pt from training
            output_path: Destination path
        """
        if model_path is None:
            # Find latest training run
            runs_dir = Path(self._output_dir) / "safewatch_yolo"
            if runs_dir.exists():
                best = runs_dir / "weights" / "best.pt"
                if best.exists():
                    model_path = str(best)

        if model_path is None or not Path(model_path).exists():
            print("⚠️ No trained model found")
            return

        import shutil
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, dest)
        print(f"✅ Model exported to: {dest}")

    def evaluate(self, model_path: str = None, data_yaml: str = None):
        """Evaluate the trained model on validation set."""
        if model_path is None:
            runs_dir = Path(self._output_dir) / "safewatch_yolo"
            if runs_dir.exists():
                model_path = str(runs_dir / "weights" / "best.pt")

        if model_path is None or not Path(model_path).exists():
            print("⚠️ No model to evaluate")
            return

        model = YOLO(model_path)
        results = model.val(data=data_yaml or self._data_yaml)

        print(f"\nValidation Results:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")

        return results


if __name__ == "__main__":
    trainer = CustomYOLOTrainer()
    trainer.prepare_dataset_yaml()
    trainer.train(epochs=100)
    trainer.export_model()
