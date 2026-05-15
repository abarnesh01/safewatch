"""
SafeWatch Detector Tests
Unit tests for AI detectors and behavioral intelligence engines.
"""

import unittest
import numpy as np
from detection.person_detector import PersonDetector
from detection.pose_estimator import PoseEstimator
from threats.fight_detector import FightDetector


class TestDetectors(unittest.TestCase):
    """Test suite for the detection and threats modules."""

    def test_person_detector_init(self):
        """Test PersonDetector loading."""
        detector = PersonDetector()
        # Note: Might be not ready if model file is missing
        self.assertIsInstance(detector, PersonDetector)

    def test_fight_detector_logic(self):
        """Test fight detection logic with mock data."""
        detector = FightDetector(proximity_threshold=100.0)
        # Mock empty detection result
        events = detector.detect("cam_01", [], {}, {})
        self.assertEqual(len(events), 0)


if __name__ == "__main__":
    unittest.main()
