"""
SafeWatch — Export ONNX
Export trained PyTorch action classifier to ONNX format for CPU inference.
"""

import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


ACTION_CLASSES = [
    "normal", "fight", "fall", "assault", "harassment",
    "abuse", "panic", "unconscious", "other",
]


class ActionLSTM(nn.Module):
    """LSTM model matching the training architecture."""

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
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.classifier(out)
        return out


class ONNXExporter:
    """Exports trained PyTorch model to ONNX format and validates it."""

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required. Install with: pip install torch")

    def __repr__(self) -> str:
        return "ONNXExporter()"

    def export(
        self,
        model_path: str = "checkpoints/best_model.pt",
        output_path: str = "models/action_classifier.onnx",
        opset_version: int = 17,
    ) -> str:
        """
        Export a trained PyTorch model to ONNX.

        Args:
            model_path: Path to the .pt model file
            output_path: Path for the output .onnx file
            opset_version: ONNX opset version

        Returns:
            Path to the exported ONNX model
        """
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        checkpoint = torch.load(str(model_file), map_location="cpu")

        model = ActionLSTM(
            input_size=99,
            hidden_size=256,
            num_layers=2,
            num_classes=len(ACTION_CLASSES),
            dropout=0.0,  # No dropout in inference
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 30, 99)

        # Export
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(out),
            input_names=["pose_sequence"],
            output_names=["action_probs"],
            dynamic_axes={
                "pose_sequence": {0: "batch_size"},
                "action_probs": {0: "batch_size"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

        file_size = out.stat().st_size / (1024 * 1024)
        print(f"✅ ONNX model exported to: {out}")
        print(f"   Model size: {file_size:.2f} MB")
        print(f"   Opset version: {opset_version}")

        return str(out)

    def validate(self, onnx_path: str = "models/action_classifier.onnx") -> bool:
        """
        Validate the exported ONNX model with onnxruntime.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            True if validation passes
        """
        if not ORT_AVAILABLE:
            print("⚠️ onnxruntime not available, skipping validation")
            return False

        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        try:
            session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])

            # Check inputs/outputs
            inputs = session.get_inputs()
            outputs = session.get_outputs()

            print(f"\nModel Inputs:")
            for inp in inputs:
                print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")

            print(f"\nModel Outputs:")
            for out in outputs:
                print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

            # Test inference
            dummy = np.random.randn(1, 30, 99).astype(np.float32)
            result = session.run(["action_probs"], {"pose_sequence": dummy})

            probs = result[0][0]
            print(f"\nTest inference output shape: {result[0].shape}")
            print(f"Test output values: {probs}")

            # Check output makes sense
            assert result[0].shape == (1, len(ACTION_CLASSES)), \
                f"Expected (1, {len(ACTION_CLASSES)}), got {result[0].shape}"

            print("✅ ONNX model validation passed!")
            return True

        except Exception as e:
            print(f"❌ ONNX validation failed: {e}")
            return False

    def benchmark(
        self,
        onnx_path: str = "models/action_classifier.onnx",
        num_runs: int = 100,
    ):
        """
        Run speed benchmark on the ONNX model.

        Args:
            onnx_path: Path to ONNX model
            num_runs: Number of inference runs to average
        """
        if not ORT_AVAILABLE:
            print("⚠️ onnxruntime not available")
            return

        session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

        dummy = np.random.randn(1, 30, 99).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(["action_probs"], {"pose_sequence": dummy})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(["action_probs"], {"pose_sequence": dummy})
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)

        # Model info
        model_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        estimated_ram = model_size * 2.5  # Rough estimate

        print(f"\n{'=' * 50}")
        print(f"ONNX Inference Benchmark ({num_runs} runs)")
        print(f"{'=' * 50}")
        print(f"Model size:     {model_size:.2f} MB")
        print(f"Estimated RAM:  {estimated_ram:.2f} MB")
        print(f"Avg latency:    {avg:.2f} ms")
        print(f"P50 latency:    {p50:.2f} ms")
        print(f"P95 latency:    {p95:.2f} ms")
        print(f"P99 latency:    {p99:.2f} ms")
        print(f"Throughput:     {1000 / avg:.0f} inferences/sec")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    exporter = ONNXExporter()
    onnx_path = exporter.export()
    exporter.validate(onnx_path)
    exporter.benchmark(onnx_path)
