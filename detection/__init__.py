"""
SafeWatch Detection Package
Provides person detection, pose estimation, optical flow, and zone management.
"""

from detection.person_detector import PersonDetector
from detection.pose_estimator import PoseEstimator
from detection.optical_flow import OpticalFlowAnalyzer
from detection.zone_manager import ZoneManager

__all__ = ["PersonDetector", "PoseEstimator", "OpticalFlowAnalyzer", "ZoneManager"]
