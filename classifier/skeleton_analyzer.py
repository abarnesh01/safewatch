"""
SafeWatch — SkeletonAnalyzer
Extracts behavioral features from pose estimation results.
"""

from typing import Optional

import numpy as np
from loguru import logger

from detection.pose_estimator import PoseResult


class SkeletonAnalyzer:
    """
    Analyzes pose landmarks to extract high-level behavioral features
    such as body orientation, arm raise levels, lean angles, and more.
    """

    def __init__(self):
        logger.info("SkeletonAnalyzer initialized")

    def __repr__(self) -> str:
        return "SkeletonAnalyzer()"

    def get_body_orientation(self, pose: PoseResult) -> Optional[str]:
        """
        Determine the body orientation of a person.

        Returns:
            One of "standing", "sitting", "lying", "crouching", or None if insufficient data.
        """
        left_hip = pose.get_landmark("left_hip")
        right_hip = pose.get_landmark("right_hip")
        left_shoulder = pose.get_landmark("left_shoulder")
        right_shoulder = pose.get_landmark("right_shoulder")
        left_knee = pose.get_landmark("left_knee")
        right_knee = pose.get_landmark("right_knee")
        left_ankle = pose.get_landmark("left_ankle")
        right_ankle = pose.get_landmark("right_ankle")

        if any(kp is None for kp in [left_hip, right_hip, left_shoulder, right_shoulder]):
            return None

        hip_y = (left_hip["y"] + right_hip["y"]) / 2
        shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        torso_height = abs(hip_y - shoulder_y)

        hip_x = (left_hip["x"] + right_hip["x"]) / 2
        shoulder_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
        torso_dx = abs(hip_x - shoulder_x)
        torso_dy = abs(hip_y - shoulder_y)

        if torso_dy < 0.01:
            return "lying"

        torso_angle = np.degrees(np.arctan2(torso_dx, torso_dy))

        if torso_angle > 60:
            return "lying"

        if left_knee is not None and right_knee is not None:
            knee_y = (left_knee["y"] + right_knee["y"]) / 2

            if left_ankle is not None and right_ankle is not None:
                ankle_y = (left_ankle["y"] + right_ankle["y"]) / 2
                knee_hip_dist = abs(knee_y - hip_y)
                knee_ankle_dist = abs(knee_y - ankle_y)

                if knee_hip_dist < 0.03 and torso_angle < 30:
                    return "sitting"

            hip_to_knee = knee_y - hip_y
            if hip_to_knee < torso_height * 0.5 and torso_angle > 20:
                return "crouching"

        if torso_angle < 30:
            return "standing"

        return "standing"

    def get_arm_raise_level(self, pose: PoseResult) -> Optional[float]:
        """
        Get how raised the arms are (average of both arms).

        Returns:
            0.0 (arms down at sides) to 1.0 (arms fully raised above head), or None.
        """
        left_shoulder = pose.get_landmark("left_shoulder")
        right_shoulder = pose.get_landmark("right_shoulder")
        left_wrist = pose.get_landmark("left_wrist")
        right_wrist = pose.get_landmark("right_wrist")
        left_hip = pose.get_landmark("left_hip")
        right_hip = pose.get_landmark("right_hip")

        levels = []

        if left_shoulder is not None and left_wrist is not None and left_hip is not None:
            hip_y = left_hip["y"]
            shoulder_y = left_shoulder["y"]
            wrist_y = left_wrist["y"]

            full_range = abs(hip_y - shoulder_y) * 2
            if full_range > 0.01:
                raise_amount = (hip_y - wrist_y) / full_range
                levels.append(max(0.0, min(1.0, raise_amount)))

        if right_shoulder is not None and right_wrist is not None and right_hip is not None:
            hip_y = right_hip["y"]
            shoulder_y = right_shoulder["y"]
            wrist_y = right_wrist["y"]

            full_range = abs(hip_y - shoulder_y) * 2
            if full_range > 0.01:
                raise_amount = (hip_y - wrist_y) / full_range
                levels.append(max(0.0, min(1.0, raise_amount)))

        if not levels:
            return None

        return float(np.mean(levels))

    def get_body_lean_angle(self, pose: PoseResult) -> Optional[float]:
        """
        Get the lean angle of the body from vertical (in degrees).

        Returns:
            Degrees from vertical (0 = perfectly upright), or None.
        """
        left_shoulder = pose.get_landmark("left_shoulder")
        right_shoulder = pose.get_landmark("right_shoulder")
        left_hip = pose.get_landmark("left_hip")
        right_hip = pose.get_landmark("right_hip")

        if any(kp is None for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        mid_shoulder_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
        mid_shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        mid_hip_x = (left_hip["x"] + right_hip["x"]) / 2
        mid_hip_y = (left_hip["y"] + right_hip["y"]) / 2

        dx = mid_shoulder_x - mid_hip_x
        dy = mid_hip_y - mid_shoulder_y

        if abs(dy) < 1e-6:
            return 90.0

        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
        return float(angle)

    def get_torso_rotation(self, pose: PoseResult) -> Optional[float]:
        """
        Get the rotation of the torso (twist) in degrees.

        Returns:
            Degrees of torso rotation, or None.
        """
        left_shoulder = pose.get_landmark("left_shoulder")
        right_shoulder = pose.get_landmark("right_shoulder")
        left_hip = pose.get_landmark("left_hip")
        right_hip = pose.get_landmark("right_hip")

        if any(kp is None for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        shoulder_dx = right_shoulder["x"] - left_shoulder["x"]
        hip_dx = right_hip["x"] - left_hip["x"]

        if abs(hip_dx) < 1e-6:
            return 0.0

        ratio = shoulder_dx / hip_dx
        ratio = max(-2.0, min(2.0, ratio))
        rotation = np.degrees(np.arctan2(abs(1 - ratio), 1))
        return float(rotation)

    def get_head_position_relative_to_hips(self, pose: PoseResult) -> Optional[float]:
        """
        Get the vertical offset of the head (nose) relative to the hips.

        Returns:
            Normalized y offset (negative = head below hips), or None.
        """
        nose = pose.get_landmark("nose")
        left_hip = pose.get_landmark("left_hip")
        right_hip = pose.get_landmark("right_hip")

        if nose is None or left_hip is None or right_hip is None:
            return None

        hip_y = (left_hip["y"] + right_hip["y"]) / 2
        offset = hip_y - nose["y"]
        return float(offset)

    def is_person_horizontal(self, pose: PoseResult, threshold: float = 25.0) -> Optional[bool]:
        """
        Check if a person's body is horizontal (lying down).

        Args:
            pose: PoseResult object
            threshold: Maximum angle from horizontal to be considered horizontal

        Returns:
            True if horizontal, False if not, None if insufficient data.
        """
        lean_angle = self.get_body_lean_angle(pose)
        if lean_angle is None:
            return None

        is_horizontal = lean_angle > (90.0 - threshold)
        return is_horizontal

    def get_center_of_mass(self, pose: PoseResult) -> Optional[tuple[float, float]]:
        """
        Estimate center of mass from major body landmarks.

        Returns:
            (x, y) normalized coordinates, or None.
        """
        major_joints = [
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
        ]

        xs, ys = [], []
        weights = [1.0, 1.0, 1.5, 1.5, 0.8, 0.8]

        for joint, weight in zip(major_joints, weights):
            kp = pose.get_landmark(joint)
            if kp is not None:
                xs.append(kp["x"] * weight)
                ys.append(kp["y"] * weight)

        if len(xs) < 3:
            return None

        total_weight = sum(weights[:len(xs)])
        com_x = sum(xs) / total_weight
        com_y = sum(ys) / total_weight
        return (float(com_x), float(com_y))

    def get_limb_extension(self, pose: PoseResult, limb: str) -> Optional[float]:
        """
        Get the extension ratio of a limb (0.0 = fully bent, 1.0 = fully extended).

        Args:
            pose: PoseResult object
            limb: One of "left_arm", "right_arm", "left_leg", "right_leg"

        Returns:
            Extension ratio 0.0 to 1.0, or None.
        """
        limb_joints = {
            "left_arm": ("left_shoulder", "left_elbow", "left_wrist"),
            "right_arm": ("right_shoulder", "right_elbow", "right_wrist"),
            "left_leg": ("left_hip", "left_knee", "left_ankle"),
            "right_leg": ("right_hip", "right_knee", "right_ankle"),
        }

        joints = limb_joints.get(limb)
        if joints is None:
            return None

        j1_name, j2_name, j3_name = joints
        j1 = pose.get_landmark(j1_name)
        j2 = pose.get_landmark(j2_name)
        j3 = pose.get_landmark(j3_name)

        if any(j is None for j in [j1, j2, j3]):
            return None

        v1 = np.array([j1["x"] - j2["x"], j1["y"] - j2["y"]])
        v2 = np.array([j3["x"] - j2["x"], j3["y"] - j2["y"]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        extension = angle / 180.0
        return float(max(0.0, min(1.0, extension)))

    def get_inter_person_distance(self, pose1: PoseResult, pose2: PoseResult) -> Optional[float]:
        """
        Calculate the normalized distance between two persons' centers of mass.

        Returns:
            Distance in normalized coordinates, or None.
        """
        com1 = self.get_center_of_mass(pose1)
        com2 = self.get_center_of_mass(pose2)

        if com1 is None or com2 is None:
            return None

        dx = com1[0] - com2[0]
        dy = com1[1] - com2[1]
        return float(np.sqrt(dx**2 + dy**2))

    def get_facing_direction(self, pose: PoseResult) -> Optional[str]:
        """
        Estimate which direction the person is facing.

        Returns:
            One of "left", "right", "forward", "backward", or None.
        """
        left_shoulder = pose.get_landmark("left_shoulder")
        right_shoulder = pose.get_landmark("right_shoulder")
        nose = pose.get_landmark("nose")

        if any(kp is None for kp in [left_shoulder, right_shoulder]):
            return None

        shoulder_width = abs(right_shoulder["x"] - left_shoulder["x"])

        if nose is not None:
            mid_shoulder_x = (left_shoulder["x"] + right_shoulder["x"]) / 2
            nose_offset = nose["x"] - mid_shoulder_x

            if abs(nose_offset) > shoulder_width * 0.3:
                return "right" if nose_offset > 0 else "left"

        if shoulder_width < 0.03:
            left_z = left_shoulder.get("z", 0)
            right_z = right_shoulder.get("z", 0)
            if abs(left_z - right_z) > 0.1:
                return "left" if left_z < right_z else "right"
            return "forward"

        return "forward"
