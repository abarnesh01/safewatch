"""
SafeWatch — PoseEstimator
MediaPipe-based pose estimation with skeleton drawing and joint utilities.
Uses the MediaPipe Tasks API (PoseLandmarker) with temporal smoothing and interpolation.
"""

import os
import threading
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Union
from collections import defaultdict, deque
from loguru import logger

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not available — pose estimation disabled")

from detection.person_detector import Person


KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
]

# Default model path relative to project root
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "pose_landmarker_lite.task",
)


@dataclass
class PoseResult:
    """Contains pose estimation results for a single person."""
    person_id: int
    landmarks: list[dict]  # 33 points with x, y, z, visibility
    keypoints: dict  # Named keypoints
    bbox: tuple[int, int, int, int]
    confidence: float

    def __repr__(self) -> str:
        n_visible = sum(1 for lm in self.landmarks if lm.get("visibility", 0) > 0.5)
        return (
            f"PoseResult(person_id={self.person_id}, "
            f"visible={n_visible}/33, conf={self.confidence:.2f})"
        )

    def get_landmark(self, name: str) -> Optional[dict]:
        """Get a landmark by name, returning None if not confident enough."""
        kp = self.keypoints.get(name)
        if kp is not None and kp.get("visibility", 0) > 0.3:
            return kp
        return None


class PoseEstimator:
    """
    MediaPipe Pose estimator with temporal smoothing and missing joint interpolation.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        self._config = config.get("detection", {})
        self._device = device
        self._min_confidence = self._config.get("pose_min_confidence", 0.5)
        self._model_path = self._config.get("pose_model_path", _DEFAULT_MODEL_PATH)
        self._lock = threading.Lock()
        self._pose = None
        
        # Temporal smoothing state
        self._history_len = self._config.get("pose_smoothing_frames", 5)
        self._landmark_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=self._history_len))

        if MP_AVAILABLE:
            if not os.path.isfile(self._model_path):
                logger.error(f"Pose model not found at {self._model_path}")
            else:
                base_options = mp.tasks.BaseOptions(model_asset_path=self._model_path)
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=self._min_confidence,
                    min_pose_presence_confidence=self._min_confidence,
                    min_tracking_confidence=self._min_confidence,
                )
                self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
                logger.info(f"PoseEstimator initialized (smoothing={self._history_len} frames)")
        else:
            logger.warning("PoseEstimator running without MediaPipe")

    def estimate(self, frame: np.ndarray, persons: list[Person]) -> list[PoseResult]:
        """Estimate poses with smoothing and interpolation."""
        if self._pose is None:
            return []

        results: list[PoseResult] = []
        h, w = frame.shape[:2]

        for person in persons:
            # 1. Prepare crop
            x1, y1, x2, y2 = person.bbox
            pad_x, pad_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0: continue

            # 2. Run MediaPipe
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            
            with self._lock:
                try:
                    pose_output = self._pose.detect(mp_image)
                except Exception as e:
                    logger.error(f"Pose error for {person.id}: {e}")
                    continue

            if not pose_output.pose_landmarks: continue
            raw_lms = pose_output.pose_landmarks[0]
            crop_h, crop_w = crop.shape[:2]

            # 3. Convert to global coordinates and filter low confidence
            current_lms = []
            for idx, lm in enumerate(raw_lms):
                abs_x, abs_y = lm.x * crop_w + cx1, lm.y * crop_h + cy1
                current_lms.append({
                    "x": abs_x / w, "y": abs_y / h, "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, 'visibility') else 1.0,
                    "abs_x": abs_x, "abs_y": abs_y,
                })

            # 4. Temporal Smoothing
            self._landmark_history[person.id].append(current_lms)
            smoothed_lms = self._apply_smoothing(person.id)
            
            # 5. Missing Joint Interpolation
            final_lms = self._interpolate_landmarks(smoothed_lms)
            
            # 6. Build Result
            keypoints = {KEYPOINT_NAMES[i]: final_lms[i] for i in range(len(KEYPOINT_NAMES))}
            avg_vis = np.mean([lm["visibility"] for lm in final_lms])

            results.append(PoseResult(
                person_id=person.id,
                landmarks=final_lms,
                keypoints=keypoints,
                bbox=person.bbox,
                confidence=float(avg_vis),
            ))

        return results

    def _apply_smoothing(self, person_id: int) -> list[dict]:
        """Average landmarks over recent history."""
        history = self._landmark_history[person_id]
        if len(history) < 2: return history[-1]

        n_lms = len(history[0])
        smoothed = []
        for i in range(n_lms):
            vals = [h[i] for h in history]
            vis = [v["visibility"] for v in vals]
            
            # Weight by visibility
            total_vis = sum(vis) + 1e-6
            avg_x = sum(v["x"] * v["visibility"] for v in vals) / total_vis
            avg_y = sum(v["y"] * v["visibility"] for v in vals) / total_vis
            avg_z = sum(v["z"] * v["visibility"] for v in vals) / total_vis
            avg_abs_x = sum(v["abs_x"] * v["visibility"] for v in vals) / total_vis
            avg_abs_y = sum(v["abs_y"] * v["visibility"] for v in vals) / total_vis

            smoothed.append({
                "x": avg_x, "y": avg_y, "z": avg_z,
                "visibility": np.mean(vis),
                "abs_x": avg_abs_x, "abs_y": avg_abs_y,
            })
        return smoothed

    def _interpolate_landmarks(self, landmarks: list[dict]) -> list[dict]:
        """Interpolate missing symmetrical joints (e.g. if one hip is hidden)."""
        pairs = [
            (11, 12), # shoulders
            (23, 24), # hips
            (13, 14), # elbows
            (15, 16), # wrists
            (25, 26), # knees
            (27, 28), # ankles
        ]
        
        for p1, p2 in pairs:
            lm1, lm2 = landmarks[p1], landmarks[p2]
            if lm1["visibility"] > 0.5 and lm2["visibility"] < 0.3:
                # Estimate lm2 based on lm1 (simple mirror for visibility/existence)
                lm2["visibility"] = lm1["visibility"] * 0.5
            elif lm2["visibility"] > 0.5 and lm1["visibility"] < 0.3:
                lm1["visibility"] = lm2["visibility"] * 0.5
        
        return landmarks

    def draw_skeleton(self, frame: np.ndarray, pose_results: list[PoseResult]) -> np.ndarray:
        """Draw stabilized skeleton overlays."""
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]
        for pose in pose_results:
            color = colors[pose.person_id % len(colors)]
            for s, e in SKELETON_CONNECTIONS:
                if s < len(pose.landmarks) and e < len(pose.landmarks):
                    lm1, lm2 = pose.landmarks[s], pose.landmarks[e]
                    if lm1["visibility"] > 0.4 and lm2["visibility"] > 0.4:
                        p1 = (int(lm1["abs_x"]), int(lm1["abs_y"]))
                        p2 = (int(lm2["abs_x"]), int(lm2["abs_y"]))
                        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
            for lm in pose.landmarks:
                if lm["visibility"] > 0.4:
                    cv2.circle(frame, (int(lm["abs_x"]), int(lm["abs_y"])), 3, color, -1)
        return frame

    def close(self):
        """Release MediaPipe resources."""
        if self._pose:
            self._pose.close()
            self._pose = None
            logger.info("PoseEstimator closed")
