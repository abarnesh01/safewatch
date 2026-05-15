"""
SafeWatch — PoseEstimator
MediaPipe-based pose estimation with skeleton drawing and joint utilities.
Uses the MediaPipe Tasks API (PoseLandmarker).
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
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

IMPORTANT_KEYPOINTS = [
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


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
            f"visible_landmarks={n_visible}/33, confidence={self.confidence:.2f})"
        )

    def get_landmark(self, name: str) -> Optional[dict]:
        """Get a landmark by name, returning None if not confident enough."""
        kp = self.keypoints.get(name)
        if kp is not None and kp.get("visibility", 0) > 0.3:
            return kp
        return None


# Default model path relative to project root
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "pose_landmarker_lite.task",
)


class PoseEstimator:
    """
    MediaPipe Pose estimator that processes cropped person bounding boxes
    to extract 33 body landmarks per person.

    Uses the MediaPipe Tasks API (PoseLandmarker) instead of the deprecated
    mp.solutions.pose interface.
    """

    def __init__(self, config: dict):
        self._config = config.get("detection", {})
        self._min_confidence = self._config.get("pose_min_confidence", 0.5)
        self._model_path = self._config.get("pose_model_path", _DEFAULT_MODEL_PATH)
        self._lock = threading.Lock()
        self._pose = None

        if MP_AVAILABLE:
            if not os.path.isfile(self._model_path):
                logger.error(
                    f"Pose model not found at {self._model_path}. "
                    "Download it with: wget -O models/pose_landmarker_lite.task "
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
                )
            else:
                base_options = mp.tasks.BaseOptions(
                    model_asset_path=self._model_path,
                )
                options = mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=self._min_confidence,
                    min_pose_presence_confidence=self._min_confidence,
                    min_tracking_confidence=self._min_confidence,
                )
                self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
                logger.info(
                    f"PoseEstimator initialized (model={os.path.basename(self._model_path)})"
                )
        else:
            logger.warning("PoseEstimator running without MediaPipe — no pose data")

    def __repr__(self) -> str:
        available = self._pose is not None
        return f"PoseEstimator(available={available}, model={os.path.basename(self._model_path)})"

    def estimate(self, frame: np.ndarray, persons: list[Person]) -> list[PoseResult]:
        """
        Estimate poses for each detected person.

        Args:
            frame: Full BGR frame
            persons: List of detected Person objects

        Returns:
            List of PoseResult objects, one per person with detected pose
        """
        if self._pose is None:
            return []

        results: list[PoseResult] = []
        h, w = frame.shape[:2]

        for person in persons:
            x1, y1, x2, y2 = person.bbox
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

            with self._lock:
                try:
                    pose_result = self._pose.detect(mp_image)
                except Exception as e:
                    logger.error(f"Pose estimation error for person {person.id}: {e}")
                    continue

            # New Tasks API: pose_landmarks is a list of NormalizedLandmarkList
            if not pose_result.pose_landmarks:
                continue

            # Take the first detected pose in this crop
            pose_lms = pose_result.pose_landmarks[0]

            landmarks = []
            keypoints = {}
            crop_h, crop_w = crop.shape[:2]

            for idx, lm in enumerate(pose_lms):
                abs_x = lm.x * crop_w + cx1
                abs_y = lm.y * crop_h + cy1

                norm_x = abs_x / w
                norm_y = abs_y / h

                landmark_data = {
                    "x": norm_x,
                    "y": norm_y,
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, 'visibility') else 1.0,
                    "abs_x": abs_x,
                    "abs_y": abs_y,
                }
                landmarks.append(landmark_data)

                if idx < len(KEYPOINT_NAMES):
                    name = KEYPOINT_NAMES[idx]
                    keypoints[name] = landmark_data

            avg_vis = np.mean([lm["visibility"] for lm in landmarks])

            results.append(PoseResult(
                person_id=person.id,
                landmarks=landmarks,
                keypoints=keypoints,
                bbox=person.bbox,
                confidence=float(avg_vis),
            ))

        return results

    def draw_skeleton(self, frame: np.ndarray, pose_results: list[PoseResult]) -> np.ndarray:
        """
        Draw skeleton lines and keypoint dots on the frame.

        Args:
            frame: BGR image to draw on
            pose_results: List of PoseResult objects

        Returns:
            Annotated frame
        """
        colors = [
            (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (0, 255, 0), (255, 128, 0), (128, 0, 255),
        ]

        for pose in pose_results:
            color = colors[pose.person_id % len(colors)]

            for start_idx, end_idx in SKELETON_CONNECTIONS:
                if start_idx >= len(pose.landmarks) or end_idx >= len(pose.landmarks):
                    continue
                lm1 = pose.landmarks[start_idx]
                lm2 = pose.landmarks[end_idx]

                if lm1["visibility"] < 0.5 or lm2["visibility"] < 0.5:
                    continue

                pt1 = (int(lm1["abs_x"]), int(lm1["abs_y"]))
                pt2 = (int(lm2["abs_x"]), int(lm2["abs_y"]))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

            for lm in pose.landmarks:
                if lm["visibility"] < 0.5:
                    continue
                pt = (int(lm["abs_x"]), int(lm["abs_y"]))
                cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)

        return frame

    @staticmethod
    def get_body_angle(pose: PoseResult, joint1: str, joint2: str, joint3: str) -> Optional[float]:
        """
        Calculate the angle at joint2 formed by joint1-joint2-joint3.

        Args:
            pose: PoseResult object
            joint1: Name of first joint
            joint2: Name of vertex joint
            joint3: Name of third joint

        Returns:
            Angle in degrees, or None if any joint has low visibility
        """
        kp1 = pose.get_landmark(joint1)
        kp2 = pose.get_landmark(joint2)
        kp3 = pose.get_landmark(joint3)

        if any(kp is None for kp in [kp1, kp2, kp3]):
            return None

        v1 = np.array([kp1["x"] - kp2["x"], kp1["y"] - kp2["y"]])
        v2 = np.array([kp3["x"] - kp2["x"], kp3["y"] - kp2["y"]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        return float(angle)

    @staticmethod
    def get_joint_velocity(
        pose: PoseResult, joint: str, previous_pose: Optional[PoseResult]
    ) -> Optional[float]:
        """
        Calculate the velocity of a joint between two consecutive pose estimations.

        Args:
            pose: Current pose
            joint: Joint name
            previous_pose: Previous pose for same person

        Returns:
            Velocity in normalized coordinate units per frame, or None
        """
        if previous_pose is None:
            return None

        curr = pose.get_landmark(joint)
        prev = previous_pose.get_landmark(joint)

        if curr is None or prev is None:
            return None

        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        return float(np.sqrt(dx**2 + dy**2))

    def close(self):
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None
            logger.info("PoseEstimator closed")
