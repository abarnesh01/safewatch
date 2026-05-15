"""
SafeWatch — Classifier Training
Trains a high-performance LSTM-based action classifier on extracted pose sequences.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class ActionLSTM(nn.Module):
    """LSTM model for action recognition from pose sequences."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(ActionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the last time step output
        out = self.fc(out[:, -1, :])
        return out

class Trainer:
    def __init__(self, data_path: str = "data/pose_sequences.npz", output_dir: str = "models"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Trainer initialized using device: {self.device}")

    def load_data(self, test_size=0.2):
        if not self.data_path.exists():
            logger.error(f"Data file not found at {self.data_path}")
            return None, None, None, None, None

        data = np.load(self.data_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        classes = data['classes']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        return X_train, X_test, y_train, y_test, classes

    def train(self, epochs: int = 50, batch_size: int = 32, lr: float = 0.001):
        X_train, X_test, y_train, y_test, classes = self.load_data()
        if X_train is None:
            return

        # Convert to tensors
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        input_size = X_train.shape[2] # 99 landmarks (33*3)
        num_classes = len(classes)
        
        model = ActionLSTM(input_size=input_size, hidden_size=256, num_layers=2, num_classes=num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        logger.info(f"Starting training for {epochs} epochs...")
        best_acc = 0.0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            acc = 100 * correct / total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            scheduler.step(avg_val_loss)

            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Acc: {acc:.2f}%")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), self.output_dir / "action_model_best.pt")
                logger.info(f"New best model saved with accuracy: {best_acc:.2f}%")

        logger.info("Training finished.")
        self.evaluate(model, test_loader, classes)

    def evaluate(self, model, test_loader, classes):
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        logger.info("\n" + classification_report(y_true, y_pred, target_names=classes))
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(epochs=10, batch_size=16) # Reduced for demo
