"""
SafeWatch — ActionClassifier
ONNX-based or rule-based action classification from pose sequences.
"""

import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import deque

import numpy as np
from loguru import logger

from detection.pose_estimator import PoseResult
from classifier.skeleton_analyzer import SkeletonAnalyzer


ACTION_CLASSES = [
    "normal", "fight", "fall", "assault", "harassment",
    "abuse", "panic", "unconscious", "other",
]


@dataclass
class ActionResult:
    """Result from action classification."""
    action_class: str
    confidence: float
    top3_predictions: list[tuple[str, float]]

    def __repr__(self) -> str:
        return (
            f"ActionResult(class='{self.action_class}', "
            f"confidence={self.confidence:.2f})"
        )


class ActionClassifier:
    """
    Action classifier that uses ONNX model when available, falling back
    to rule-based classification using skeleton features and velocities.
    """

    def __init__(self, config: dict):
        self._config = config
        model_config = config.get("models", {})
        self._model_path = Path(model_config.get("action_classifier", "models/action_classifier.onnx"))
        self._session = None
        self._lock = threading.Lock()
        self._skeleton_analyzer = SkeletonAnalyzer()
        self._pose_buffers: dict[int, deque] = {}
        self._buffer_size = 30
        self._use_onnx = False

        self._load_model()

    def __repr__(self) -> str:
        mode = "ONNX" if self._use_onnx else "rule-based"
        return f"ActionClassifier(mode={mode}, buffer_size={self._buffer_size})"

    def _load_model(self):
        """Attempt to load the ONNX model."""
        if not self._model_path.exists():
            logger.warning(
                f"ONNX model not found at {self._model_path} — using rule-based fallback"
            )
            return

        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(self._model_path),
                providers=["CPUExecutionProvider"],
            )
            self._use_onnx = True
            logger.info(f"ONNX action classifier loaded from {self._model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e} — using rule-based fallback")
            self._session = None
            self._use_onnx = False

    def update_buffer(self, person_id: int, pose: PoseResult):
        """
        Add a pose frame to the person's buffer.

        Args:
            person_id: Person to update
            pose: Current PoseResult
        """
        with self._lock:
            if person_id not in self._pose_buffers:
                self._pose_buffers[person_id] = deque(maxlen=self._buffer_size)
            self._pose_buffers[person_id].append(pose)

    def prepare_input(self, pose_sequence: list[PoseResult]) -> np.ndarray:
        """
        Normalize landmark coordinates into model input format.

        Args:
            pose_sequence: Sequence of PoseResult objects (up to 30 frames)

        Returns:
            numpy array of shape (1, 30, 99) — 33 landmarks × 3 coords
        """
        seq_len = self._buffer_size
        feature_dim = 99  # 33 landmarks × 3 (x, y, visibility)

        data = np.zeros((seq_len, feature_dim), dtype=np.float32)

        for t, pose in enumerate(pose_sequence[-seq_len:]):
            for i, lm in enumerate(pose.landmarks[:33]):
                base = i * 3
                data[t, base] = lm.get("x", 0.0)
                data[t, base + 1] = lm.get("y", 0.0)
                data[t, base + 2] = lm.get("visibility", 0.0)

        return data.reshape(1, seq_len, feature_dim)

    def classify(
        self,
        pose_sequence: Optional[list[PoseResult]] = None,
        skeleton_features: Optional[dict] = None,
        person_id: Optional[int] = None,
    ) -> ActionResult:
        """
        Classify the action from a pose sequence.

        Args:
            pose_sequence: List of PoseResult objects over time
            skeleton_features: Optional pre-computed features
            person_id: Optional person ID to use internal buffer

        Returns:
            ActionResult with predicted class and confidence
        """
        if pose_sequence is None and person_id is not None:
            with self._lock:
                buffer = self._pose_buffers.get(person_id)
                if buffer and len(buffer) > 0:
                    pose_sequence = list(buffer)

        if pose_sequence is None or len(pose_sequence) == 0:
            return ActionResult(
                action_class="normal",
                confidence=0.5,
                top3_predictions=[("normal", 0.5), ("other", 0.3), ("fall", 0.2)],
            )

        if self._use_onnx and self._session is not None:
            return self._classify_onnx(pose_sequence)
        else:
            return self._classify_rule_based(pose_sequence, skeleton_features)

    def _classify_onnx(self, pose_sequence: list[PoseResult]) -> ActionResult:
        """Run ONNX model inference."""
        try:
            input_data = self.prepare_input(pose_sequence)
            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name

            results = self._session.run([output_name], {input_name: input_data})
            probs = results[0][0]

            from scipy.special import softmax
            probs = softmax(probs)

            top_indices = np.argsort(probs)[::-1]
            top3 = [(ACTION_CLASSES[i], float(probs[i])) for i in top_indices[:3]]

            return ActionResult(
                action_class=ACTION_CLASSES[top_indices[0]],
                confidence=float(probs[top_indices[0]]),
                top3_predictions=top3,
            )
        except Exception as e:
            logger.error(f"ONNX inference failed: {e} — falling back to rules")
            return self._classify_rule_based(pose_sequence, None)

    def _classify_rule_based(
        self,
        pose_sequence: list[PoseResult],
        skeleton_features: Optional[dict],
    ) -> ActionResult:
        """
        Rule-based fallback classification using skeleton geometry and velocity.
        """
        scores = {cls: 0.0 for cls in ACTION_CLASSES}
        scores["normal"] = 0.4

        latest_pose = pose_sequence[-1]
        analyzer = self._skeleton_analyzer

        orientation = analyzer.get_body_orientation(latest_pose)
        arm_level = analyzer.get_arm_raise_level(latest_pose)
        lean = analyzer.get_body_lean_angle(latest_pose)
        is_horizontal = analyzer.is_person_horizontal(latest_pose)

        if is_horizontal is True:
            scores["fall"] += 0.4
            scores["unconscious"] += 0.3

            if len(pose_sequence) >= 10:
                prev_orientations = []
                for p in pose_sequence[-10:]:
                    o = analyzer.get_body_orientation(p)
                    prev_orientations.append(o)

                was_standing = any(o == "standing" for o in prev_orientations[:5])
                if was_standing:
                    scores["fall"] += 0.3

            stillness = self._check_stillness(pose_sequence[-30:])
            if stillness > 0.8:
                scores["unconscious"] += 0.3
                scores["fall"] -= 0.1

        if arm_level is not None and arm_level > 0.6:
            scores["fight"] += 0.2
            scores["assault"] += 0.15
            scores["panic"] += 0.1

        if lean is not None and lean > 40:
            scores["fall"] += 0.2
            scores["fight"] += 0.1

        if orientation == "crouching":
            scores["harassment"] += 0.15
            scores["abuse"] += 0.1

        if skeleton_features is not None:
            wrist_vel = skeleton_features.get("wrist_velocity", 0)
            if wrist_vel > 50:
                scores["fight"] += 0.3
                scores["assault"] += 0.25

            closing_speed = skeleton_features.get("closing_speed", 0)
            if closing_speed > 30:
                scores["fight"] += 0.2
                scores["assault"] += 0.15

        total = sum(scores.values())
        if total > 0:
            for cls in scores:
                scores[cls] /= total

        sorted_classes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [(cls, round(s, 3)) for cls, s in sorted_classes[:3]]

        return ActionResult(
            action_class=sorted_classes[0][0],
            confidence=round(sorted_classes[0][1], 3),
            top3_predictions=top3,
        )

    def _check_stillness(self, recent_poses: list[PoseResult]) -> float:
        """
        Check how still a person has been over recent poses.

        Returns:
            Stillness score 0.0 (moving) to 1.0 (completely still).
        """
        if len(recent_poses) < 2:
            return 0.0

        total_movement = 0.0
        count = 0

        for i in range(1, len(recent_poses)):
            curr = recent_poses[i]
            prev = recent_poses[i-1]

            for joint in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]:
                c = curr.get_landmark(joint)
                p = prev.get_landmark(joint)
                if c is not None and p is not None:
                    dx = c["x"] - p["x"]
                    dy = c["y"] - p["y"]
                    total_movement += np.sqrt(dx**2 + dy**2)
                    count += 1

        if count == 0:
            return 0.0

        avg_movement = total_movement / count
        stillness = max(0.0, 1.0 - (avg_movement * 100))
        return float(min(1.0, stillness))

    def cleanup_buffer(self, person_id: int):
        """Remove a person's pose buffer."""
        with self._lock:
            if person_id in self._pose_buffers:
                del self._pose_buffers[person_id]
