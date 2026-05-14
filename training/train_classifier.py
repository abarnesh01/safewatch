"""
SafeWatch — Train Action Classifier
LSTM-based action classifier training script for Google Colab.
"""

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


ACTION_CLASSES = [
    "normal", "fight", "fall", "assault", "harassment",
    "abuse", "panic", "unconscious", "other",
]


class PoseSequenceDataset(Dataset):
    """PyTorch dataset for pose sequence classification."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def __repr__(self) -> str:
        return f"PoseSequenceDataset(samples={len(self)}, shape={self.X.shape})"


class ActionLSTM(nn.Module):
    """LSTM-based action classification model."""

    def __init__(
        self,
        input_size: int = 99,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # Take the last time step
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        return f"ActionLSTM(params={total_params:,}, hidden={self.hidden_size}, layers={self.num_layers})"


class ActionClassifierTrainer:
    """
    Trainer for the LSTM-based action classifier.
    Designed for Google Colab with GPU support.
    """

    def __init__(
        self,
        data_path: str = "data/pose_sequences.npz",
        checkpoint_dir: str = "checkpoints",
        batch_size: int = 32,
        lr: float = 0.001,
        epochs: int = 50,
        patience: int = 10,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch torchvision")

        self._data_path = Path(data_path)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._batch_size = batch_size
        self._lr = lr
        self._epochs = epochs
        self._patience = patience

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[ActionLSTM] = None
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._train_accs: list[float] = []
        self._val_accs: list[float] = []

        print(f"Device: {self._device}")

    def __repr__(self) -> str:
        return (
            f"ActionClassifierTrainer(device={self._device}, "
            f"batch_size={self._batch_size}, lr={self._lr})"
        )

    def _load_data(self):
        """Load the preprocessed pose sequence dataset."""
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self._data_path}\n"
                "Run DatasetPrep.extract_pose_sequences() first."
            )

        data = np.load(str(self._data_path))
        X = data["X"]  # shape: (N, 30, 99)
        y = data["y"]  # shape: (N,)

        print(f"Loaded dataset: X={X.shape}, y={y.shape}")
        print(f"Classes: {np.unique(y)}")

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            cls_name = ACTION_CLASSES[cls_idx] if cls_idx < len(ACTION_CLASSES) else f"class_{cls_idx}"
            print(f"  {cls_name}: {count} samples")

        return X, y

    def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute inverse frequency class weights for imbalanced data."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        weights = np.ones(len(ACTION_CLASSES), dtype=np.float32)

        for cls_idx, count in zip(unique, counts):
            if cls_idx < len(weights):
                weights[cls_idx] = total / (len(unique) * count)

        # Normalize
        weights = weights / weights.sum() * len(ACTION_CLASSES)
        print(f"Class weights: {weights}")
        return torch.FloatTensor(weights)

    def train(self, epochs: Optional[int] = None):
        """
        Train the action classifier model.

        Args:
            epochs: Override number of training epochs
        """
        if epochs is not None:
            self._epochs = epochs

        X, y = self._load_data()

        # Split into train/val (80/20)
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(indices))
        train_idx = indices[:split]
        val_idx = indices[split:]

        train_dataset = PoseSequenceDataset(X[train_idx], y[train_idx])
        val_dataset = PoseSequenceDataset(X[val_idx], y[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

        # Model
        self._model = ActionLSTM(
            input_size=99,
            hidden_size=256,
            num_layers=2,
            num_classes=len(ACTION_CLASSES),
            dropout=0.3,
        ).to(self._device)
        print(self._model)

        # Loss with class weights
        class_weights = self._compute_class_weights(y[train_idx]).to(self._device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer + scheduler
        optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self._epochs)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self._epochs):
            start_time = time.time()

            # Train phase
            self._model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_loss /= train_total
            train_acc = train_correct / train_total

            # Validation phase
            self._model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self._device)
                    batch_y = batch_y.to(self._device)

                    outputs = self._model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * batch_X.size(0)
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()

            val_loss /= val_total
            val_acc = val_correct / val_total

            scheduler.step()
            elapsed = time.time() - start_time

            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)
            self._train_accs.append(train_acc)
            self._val_accs.append(val_acc)

            print(
                f"Epoch {epoch+1}/{self._epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Early stopping + checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = self._checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "classes": ACTION_CLASSES,
                }, str(checkpoint_path))
                print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    print(f"⏹ Early stopping at epoch {epoch+1}")
                    break

        # Save final model
        final_path = self._checkpoint_dir / "action_classifier.pt"
        torch.save(self._model.state_dict(), str(final_path))
        print(f"\n✅ Training complete. Model saved to {final_path}")

    def evaluate(self):
        """Evaluate the model on the validation set and print metrics."""
        if not SKLEARN_AVAILABLE:
            print("⚠️ scikit-learn required for evaluation metrics")
            return

        if self._model is None:
            print("⚠️ No model loaded. Train first or load a checkpoint.")
            return

        X, y = self._load_data()
        indices = np.random.permutation(len(X))
        split = int(0.8 * len(indices))
        val_idx = indices[split:]

        val_dataset = PoseSequenceDataset(X[val_idx], y[val_idx])
        val_loader = DataLoader(val_dataset, batch_size=self._batch_size, shuffle=False)

        self._model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self._device)
                outputs = self._model(batch_X)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())

        present_classes = sorted(set(all_labels))
        target_names = [ACTION_CLASSES[i] for i in present_classes]

        print("\n" + "=" * 60)
        print("Classification Report")
        print("=" * 60)
        print(classification_report(all_labels, all_preds, labels=present_classes, target_names=target_names))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=present_classes)
        print("\nConfusion Matrix:")
        print(cm)

        # Per-class F1
        f1s = f1_score(all_labels, all_preds, labels=present_classes, average=None)
        print("\nPer-class F1 scores:")
        for cls, f1 in zip(target_names, f1s):
            print(f"  {cls}: {f1:.4f}")

    def plot_curves(self, save_path: str = "training_curves.png"):
        """Plot and save training curves."""
        if not SKLEARN_AVAILABLE:
            print("⚠️ matplotlib required for plotting")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        ax1.plot(self._train_losses, label="Train Loss", color="blue")
        ax1.plot(self._val_losses, label="Val Loss", color="red")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2.plot(self._train_accs, label="Train Acc", color="blue")
        ax2.plot(self._val_accs, label="Val Acc", color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training & Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"✅ Training curves saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str = "checkpoints/best_model.pt"):
        """Load a model checkpoint."""
        cp = torch.load(checkpoint_path, map_location=self._device)

        self._model = ActionLSTM(
            input_size=99,
            hidden_size=256,
            num_layers=2,
            num_classes=len(ACTION_CLASSES),
            dropout=0.3,
        ).to(self._device)

        self._model.load_state_dict(cp["model_state_dict"])
        print(f"✅ Loaded checkpoint from epoch {cp.get('epoch', '?')} "
              f"(val_acc={cp.get('val_acc', 0):.4f})")


if __name__ == "__main__":
    trainer = ActionClassifierTrainer()
    trainer.train(epochs=50)
    trainer.evaluate()
    trainer.plot_curves()
