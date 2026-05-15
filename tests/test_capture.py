"""
SafeWatch Capture Tests
Unit tests for CameraStream and StreamManager.
"""

import unittest
import numpy as np
from capture.camera_stream import CameraStream, FramePacket
from capture.stream_manager import StreamManager


class TestCapture(unittest.TestCase):
    """Test suite for the capture module."""

    def test_camera_stream_init(self):
        """Test basic initialization of CameraStream."""
        cam = CameraStream(camera_id="test_cam", source=0)
        self.assertEqual(cam.camera_id, "test_cam")
        self.assertFalse(cam.is_running)

    def test_stream_manager_registration(self):
        """Test camera registration in StreamManager."""
        manager = StreamManager()
        manager.add_camera(camera_id="cam_01", source=0, camera_name="Test Camera")
        self.assertIn("cam_01", manager.get_camera_ids())
        self.assertEqual(manager.get_camera_count(), 1)


if __name__ == "__main__":
    unittest.main()
