"""
SafeWatch — Detector Module Tests
Tests for PersonDetector, PoseEstimator, SkeletonAnalyzer, VelocityTracker, and threat detectors.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestSkeletonAnalyzer(unittest.TestCase):
    """Tests for the SkeletonAnalyzer class."""

    def _make_pose(self, landmarks_dict: dict):
        """Helper to create a mock PoseResult."""
        from detection.pose_estimator import PoseResult

        landmarks = []
        keypoints = {}

        # Create 33 landmarks with defaults
        for i in range(33):
            lm = {"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9, "abs_x": 320, "abs_y": 240}
            landmarks.append(lm)

        # Override with provided values
        from detection.pose_estimator import KEYPOINT_NAMES
        for name, values in landmarks_dict.items():
            if name in KEYPOINT_NAMES:
                idx = KEYPOINT_NAMES.index(name)
                landmarks[idx].update(values)

        for i, name in enumerate(KEYPOINT_NAMES):
            if i < len(landmarks):
                keypoints[name] = landmarks[i]

        return PoseResult(
            person_id=1,
            landmarks=landmarks,
            keypoints=keypoints,
            bbox=(100, 100, 300, 400),
            confidence=0.9,
        )

    def test_body_orientation_standing(self):
        """Test detection of standing orientation."""
        from classifier.skeleton_analyzer import SkeletonAnalyzer

        analyzer = SkeletonAnalyzer()
        pose = self._make_pose({
            "left_shoulder": {"x": 0.45, "y": 0.3},
            "right_shoulder": {"x": 0.55, "y": 0.3},
            "left_hip": {"x": 0.45, "y": 0.6},
            "right_hip": {"x": 0.55, "y": 0.6},
            "left_knee": {"x": 0.45, "y": 0.75},
            "right_knee": {"x": 0.55, "y": 0.75},
            "left_ankle": {"x": 0.45, "y": 0.9},
            "right_ankle": {"x": 0.55, "y": 0.9},
        })
        orientation = analyzer.get_body_orientation(pose)
        self.assertEqual(orientation, "standing")

    def test_body_orientation_lying(self):
        """Test detection of lying orientation."""
        from classifier.skeleton_analyzer import SkeletonAnalyzer

        analyzer = SkeletonAnalyzer()
        pose = self._make_pose({
            "left_shoulder": {"x": 0.2, "y": 0.5},
            "right_shoulder": {"x": 0.3, "y": 0.5},
            "left_hip": {"x": 0.6, "y": 0.5},
            "right_hip": {"x": 0.7, "y": 0.5},
        })
        orientation = analyzer.get_body_orientation(pose)
        self.assertEqual(orientation, "lying")

    def test_arm_raise_level(self):
        """Test arm raise level calculation."""
        from classifier.skeleton_analyzer import SkeletonAnalyzer

        analyzer = SkeletonAnalyzer()
        pose = self._make_pose({
            "left_shoulder": {"x": 0.4, "y": 0.4},
            "right_shoulder": {"x": 0.6, "y": 0.4},
            "left_wrist": {"x": 0.3, "y": 0.2},  # Raised
            "right_wrist": {"x": 0.7, "y": 0.2},  # Raised
            "left_hip": {"x": 0.4, "y": 0.6},
            "right_hip": {"x": 0.6, "y": 0.6},
        })
        arm_level = analyzer.get_arm_raise_level(pose)
        self.assertIsNotNone(arm_level)
        self.assertGreater(arm_level, 0.5)  # Arms are raised

    def test_is_person_horizontal(self):
        """Test horizontal person detection."""
        from classifier.skeleton_analyzer import SkeletonAnalyzer

        analyzer = SkeletonAnalyzer()
        # Horizontal person
        pose = self._make_pose({
            "left_shoulder": {"x": 0.2, "y": 0.5},
            "right_shoulder": {"x": 0.3, "y": 0.5},
            "left_hip": {"x": 0.7, "y": 0.5},
            "right_hip": {"x": 0.8, "y": 0.5},
        })
        is_horizontal = analyzer.is_person_horizontal(pose)
        self.assertTrue(is_horizontal)


class TestVelocityTracker(unittest.TestCase):
    """Tests for the VelocityTracker class."""

    def test_velocity_unknown_person(self):
        """Test that unknown person ID returns 0."""
        from classifier.velocity_tracker import VelocityTracker

        tracker = VelocityTracker()
        vel = tracker.get_velocity(999, "left_wrist")
        self.assertEqual(vel, 0.0)

    def test_velocity_tracking(self):
        """Test velocity calculation with two updates."""
        from classifier.velocity_tracker import VelocityTracker
        from detection.pose_estimator import PoseResult, KEYPOINT_NAMES

        tracker = VelocityTracker()

        def make_pose(x_offset):
            landmarks = []
            keypoints = {}
            for i in range(33):
                lm = {"x": 0.5 + x_offset, "y": 0.5, "z": 0, "visibility": 0.9,
                       "abs_x": 320 + x_offset * 640, "abs_y": 240}
                landmarks.append(lm)
            for i, name in enumerate(KEYPOINT_NAMES):
                if i < len(landmarks):
                    keypoints[name] = landmarks[i]
            return PoseResult(person_id=1, landmarks=landmarks, keypoints=keypoints,
                            bbox=(100, 100, 300, 400), confidence=0.9)

        pose1 = make_pose(0.0)
        pose2 = make_pose(0.1)

        tracker.update(1, pose1, 1.0)
        tracker.update(1, pose2, 2.0)

        vel = tracker.get_velocity(1, "left_hip")
        self.assertGreater(vel, 0.0)

    def test_relative_velocity(self):
        """Test relative velocity returns 0 for unknown IDs."""
        from classifier.velocity_tracker import VelocityTracker

        tracker = VelocityTracker()
        rel_vel = tracker.get_relative_velocity(1, 2)
        self.assertEqual(rel_vel, 0.0)


class TestFightDetector(unittest.TestCase):
    """Tests for the FightDetector class."""

    def test_fight_detector_no_persons(self):
        """Test that no threats are returned with insufficient persons."""
        from threats.fight_detector import FightDetector
        from classifier.velocity_tracker import VelocityTracker

        config = {"threats": {"fight": {"enabled": True, "min_persons": 2,
                                         "confidence_threshold": 0.82,
                                         "velocity_threshold": 45.0,
                                         "overlap_threshold": 0.3}}}
        detector = FightDetector(config)
        tracker = VelocityTracker()

        # Zero persons
        events = detector.detect([], [], tracker)
        self.assertEqual(events, [])

    def test_fight_detector_disabled(self):
        """Test that disabled detector returns empty."""
        from threats.fight_detector import FightDetector
        from classifier.velocity_tracker import VelocityTracker

        config = {"threats": {"fight": {"enabled": False}}}
        detector = FightDetector(config)
        tracker = VelocityTracker()
        events = detector.detect([], [], tracker)
        self.assertEqual(events, [])

    def test_threat_event_repr(self):
        """Test ThreatEvent string representation."""
        from threats.fight_detector import ThreatEvent

        event = ThreatEvent(
            threat_type="FIGHT",
            confidence=0.9,
            persons_involved=[1, 2],
            location_bbox=(0, 0, 100, 100),
            description="Test",
            severity="HIGH",
        )
        self.assertIn("FIGHT", repr(event))
        self.assertIn("0.9", repr(event))


class TestFallDetector(unittest.TestCase):
    """Tests for the FallDetector class."""

    def test_fall_detector_no_detections(self):
        """Test with no persons detected."""
        from threats.fall_detector import FallDetector
        from classifier.velocity_tracker import VelocityTracker

        config = {"threats": {"fall": {"enabled": True, "confidence_threshold": 0.78,
                                       "hip_drop_threshold": 80, "stillness_frames": 30}}}
        detector = FallDetector(config)
        tracker = VelocityTracker()
        events = detector.detect([], [], tracker)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
