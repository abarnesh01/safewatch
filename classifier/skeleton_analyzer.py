"""
SafeWatch Skeleton Analyzer
Extracts geometric features and angles from human pose landmarks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from loguru import logger
from detection.pose_estimator import PersonPose, PoseLandmark


class SkeletonAnalyzer:
    """Analyzes skeleton geometry to extract behavioral features."""

    def __init__(self) -> None:
        logger.info("SkeletonAnalyzer initialized")

    def get_joint_angle(self, p1: PoseLandmark, p2: PoseLandmark, p3: PoseLandmark) -> float:
        """Calculate the angle between three points in degrees."""
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return float(np.degrees(angle))

    def analyze(self, pose: PersonPose) -> Dict[str, float]:
        """Extract a suite of geometric features from a pose."""
        features = {}
        lms = pose.landmarks

        try:
            # Arm angles
            if all(k in lms for k in ["left_shoulder", "left_elbow", "left_wrist"]):
                features["left_arm_angle"] = self.get_joint_angle(
                    lms["left_shoulder"], lms["left_elbow"], lms["left_wrist"]
                )
            
            if all(k in lms for k in ["right_shoulder", "right_elbow", "right_wrist"]):
                features["right_arm_angle"] = self.get_joint_angle(
                    lms["right_shoulder"], lms["right_elbow"], lms["right_wrist"]
                )

            # Leg angles
            if all(k in lms for k in ["left_hip", "left_knee", "left_ankle"]):
                features["left_leg_angle"] = self.get_joint_angle(
                    lms["left_hip"], lms["left_knee"], lms["left_ankle"]
                )

            if all(k in lms for k in ["right_hip", "right_knee", "right_ankle"]):
                features["right_leg_angle"] = self.get_joint_angle(
                    lms["right_hip"], lms["right_knee"], lms["right_ankle"]
                )

            # Body orientation
            if "left_shoulder" in lms and "right_shoulder" in lms:
                dx = lms["right_shoulder"].x - lms["left_shoulder"].x
                dy = lms["right_shoulder"].y - lms["left_shoulder"].y
                features["shoulder_tilt"] = float(np.degrees(np.arctan2(dy, dx)))

            if "left_hip" in lms and "right_hip" in lms:
                dx = lms["right_hip"].x - lms["left_hip"].x
                dy = lms["right_hip"].y - lms["left_hip"].y
                features["hip_tilt"] = float(np.degrees(np.arctan2(dy, dx)))

            # Verticality (aspect ratio of key points)
            if "nose" in lms and "left_ankle" in lms and "right_ankle" in lms:
                head_y = lms["nose"].y
                foot_y = (lms["left_ankle"].y + lms["right_ankle"].y) / 2
                height = abs(foot_y - head_y)
                
                # Shoulder width as reference
                width = 1.0
                if "left_shoulder" in lms and "right_shoulder" in lms:
                    width = np.sqrt((lms["right_shoulder"].x - lms["left_shoulder"].x)**2 + 
                                    (lms["right_shoulder"].y - lms["left_shoulder"].y)**2)
                
                features["vertical_ratio"] = height / (width + 1e-6)

            # Arm height relative to shoulders
            if "nose" in lms and "left_shoulder" in lms and "left_wrist" in lms:
                features["left_hand_raised"] = float(lms["left_wrist"].y < lms["left_shoulder"].y)
            
            if "nose" in lms and "right_shoulder" in lms and "right_wrist" in lms:
                features["right_hand_raised"] = float(lms["right_wrist"].y < lms["right_shoulder"].y)

        except Exception as exc:
            logger.error("Skeleton analysis error: {}", exc)

        return features
