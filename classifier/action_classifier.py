"""
SafeWatch Action Classifier
Classifies human actions using skeletal features and ONNX-optimized models.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from loguru import logger

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not installed, action classification limited")


class ActionClassifier:
    """Classifies temporal sequences of pose features into discrete actions."""

    ACTIONS = [
        "standing", "walking", "running", "falling", 
        "punching", "kicking", "crouching", "lying_down"
    ]

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._session = None
        self._initialized = False
        self._load_model()

    def _load_model(self) -> None:
        if not ONNX_AVAILABLE or not self._model_path or not self._model_path.exists():
            logger.info("ActionClassifier initialized in rule-based mode (no ONNX model)")
            return

        try:
            self._session = ort.InferenceSession(str(self._model_path), providers=['CPUExecutionProvider'])
            self._initialized = True
            logger.info("ActionClassifier ONNX model loaded: {}", self._model_path)
        except Exception as exc:
            logger.error("Failed to load ONNX model: {}", exc)

    def classify(self, features: Dict[str, float], 
                 velocities: Dict[str, float]) -> Tuple[str, float]:
        """Classify action based on current features and velocities."""
        
        # 1. Try ML-based classification if model is available
        if self._initialized and self._session:
            return self._predict_ml(features, velocities)
        
        # 2. Fallback to rule-based heuristic classification
        return self._predict_heuristic(features, velocities)

    def _predict_ml(self, features: Dict[str, float], 
                    velocities: Dict[str, float]) -> Tuple[str, float]:
        """Inference using the ONNX model."""
        # Placeholder for actual model input preparation
        # In production, this would use a sliding window of features
        return "unknown", 0.0

    def _predict_heuristic(self, features: Dict[str, float], 
                           velocities: Dict[str, float]) -> Tuple[str, float]:
        """Heuristic-based action classification logic."""
        
        v_person = velocities.get("person_velocity", 0.0)
        v_hands = max(velocities.get("left_wrist_velocity", 0.0), 
                      velocities.get("right_wrist_velocity", 0.0))
        
        v_ratio = features.get("vertical_ratio", 2.0)
        hands_raised = features.get("left_hand_raised", 0) or features.get("right_hand_raised", 0)

        # Falling Detection
        if v_ratio < 0.8 and v_person > 10.0:
            return "falling", 0.85
        
        if v_ratio < 0.5:
            return "lying_down", 0.90

        # Aggression Detection
        if v_hands > 20.0 and hands_raised:
            return "punching", 0.75
        
        # Movement Detection
        if v_person > 15.0:
            return "running", 0.80
        elif v_person > 3.0:
            return "walking", 0.80
        
        return "standing", 0.95

    @property
    def is_ml_enabled(self) -> bool:
        return self._initialized
