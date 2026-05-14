"""
SafeWatch Pose Estimator
MediaPipe-based pose estimation for human behavior analysis.
"""

import cv2
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass

from loguru import logger

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logger.warning("MediaPipe not installed, pose estimation disabled")


@dataclass
class PoseLandmark:
    """Named joint landmark with coordinates and visibility."""
    name: str
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PersonPose:
    """Complete pose estimation for a person."""
    person_id: int
    landmarks: Dict[str, PoseLandmark]
    world_landmarks: Dict[str, PoseLandmark]
    bbox: tuple  # (x1, y1, x2, y2) relative to original frame


class PoseEstimator:
    """High-performance pose estimator using MediaPipe."""

    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    def __init__(self, model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        self._complexity = model_complexity
        self._min_detection_conf = min_detection_confidence
        self._min_tracking_conf = min_tracking_confidence
        self._pose_engine = None
        self._initialized = False
        self._initialize()

    def _initialize(self) -> None:
        if not MP_AVAILABLE:
            return
        try:
            self._mp_pose = mp.solutions.pose
            self._pose_engine = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self._complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self._min_detection_conf,
                min_tracking_confidence=self._min_tracking_conf
            )
            self._initialized = True
            logger.info("MediaPipe Pose Estimator initialized (complexity={})", self._complexity)
        except Exception as exc:
            logger.error("Failed to initialize MediaPipe Pose: {}", exc)
            self._initialized = False

    def estimate(self, frame: np.ndarray, person_id: int = 0, 
                 bbox: Optional[tuple] = None) -> Optional[PersonPose]:
        """Estimate pose for a single person within a bounding box."""
        if not self._initialized or self._pose_engine is None:
            return None

        try:
            # If bbox is provided, crop the frame for better accuracy
            input_frame = frame
            x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
            
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                # Add padding
                pad_w = int((x2 - x1) * 0.2)
                pad_h = int((y2 - y1) * 0.2)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(frame.shape[1], x2 + pad_w)
                y2 = min(frame.shape[0], y2 + pad_h)
                
                if x2 > x1 and y2 > y1:
                    input_frame = frame[y1:y2, x1:x2]
                else:
                    return None

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            results = self._pose_engine.process(rgb_frame)

            if not results.pose_landmarks:
                return None

            landmarks = {}
            world_landmarks = {}
            
            h, w = input_frame.shape[:2]
            
            for i, name in enumerate(self.LANDMARK_NAMES):
                lm = results.pose_landmarks.landmark[i]
                # Map back to original frame coordinates
                abs_x = (lm.x * w) + x1
                abs_y = (lm.y * h) + y1
                
                landmarks[name] = PoseLandmark(
                    name=name, x=abs_x, y=abs_y, z=lm.z, visibility=lm.visibility
                )
                
                if results.pose_world_landmarks:
                    wlm = results.pose_world_landmarks.landmark[i]
                    world_landmarks[name] = PoseLandmark(
                        name=name, x=wlm.x, y=wlm.y, z=wlm.z, visibility=wlm.visibility
                    )

            return PersonPose(
                person_id=person_id,
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                bbox=(x1, y1, x2, y2)
            )

        except Exception as exc:
            logger.error("Pose estimation error for person {}: {}", person_id, exc)
            return None

    def draw_pose(self, frame: np.ndarray, pose: PersonPose, 
                  color: tuple = (0, 255, 255)) -> np.ndarray:
        """Render skeleton onto the frame."""
        annotated = frame.copy()
        
        # Define skeleton connections
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"), ("right_knee", "right_ankle")
        ]

        # Draw landmarks
        for name, lm in pose.landmarks.items():
            if lm.visibility > 0.5:
                cv2.circle(annotated, (int(lm.x), int(lm.y)), 4, color, -1)

        # Draw connections
        for start_name, end_name in connections:
            if start_name in pose.landmarks and end_name in pose.landmarks:
                s = pose.landmarks[start_name]
                e = pose.landmarks[end_name]
                if s.visibility > 0.5 and e.visibility > 0.5:
                    cv2.line(annotated, (int(s.x), int(s.y)), (int(e.x), int(e.y)), color, 2)

        return annotated

    def close(self) -> None:
        if self._pose_engine:
            self._pose_engine.close()
            logger.info("Pose Estimator closed")
