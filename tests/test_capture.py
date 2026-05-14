"""
SafeWatch — Capture Module Tests
Tests for CameraStream, FrameSampler, and StreamManager.
"""

import time
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class TestCameraStream(unittest.TestCase):
    """Tests for the CameraStream class."""

    def test_camera_stream_creation(self):
        """Test CameraStream can be instantiated with valid config."""
        from capture.camera_stream import CameraStream

        stream = CameraStream(
            camera_id="TEST-01",
            source=0,
            resolution=(640, 480),
            fps_target=15,
            name="Test Camera",
        )
        self.assertEqual(stream.camera_id, "TEST-01")
        self.assertEqual(stream.name, "Test Camera")
        self.assertFalse(stream.is_running())
        self.assertFalse(stream.is_connected())

    def test_camera_stream_status(self):
        """Test status reporting."""
        from capture.camera_stream import CameraStream

        stream = CameraStream(
            camera_id="TEST-02",
            source=0,
            resolution=(320, 240),
        )
        status = stream.get_status()
        self.assertEqual(status["camera_id"], "TEST-02")
        self.assertFalse(status["running"])
        self.assertFalse(status["connected"])
        self.assertEqual(status["resolution"], (320, 240))

    def test_camera_stream_repr(self):
        """Test string representation."""
        from capture.camera_stream import CameraStream

        stream = CameraStream(camera_id="CAM-01", source=0)
        repr_str = repr(stream)
        self.assertIn("CAM-01", repr_str)
        self.assertIn("stopped", repr_str)

    def test_read_returns_none_when_empty(self):
        """Test that read() returns None when buffer is empty."""
        from capture.camera_stream import CameraStream

        stream = CameraStream(camera_id="CAM-TEST", source=0)
        result = stream.read()
        self.assertIsNone(result)

    def test_fps_initial_zero(self):
        """Test that FPS starts at 0."""
        from capture.camera_stream import CameraStream

        stream = CameraStream(camera_id="CAM-FPS", source=0)
        self.assertEqual(stream.get_fps(), 0.0)


class TestFrameSampler(unittest.TestCase):
    """Tests for the FrameSampler class."""

    def test_frame_sampler_creation(self):
        """Test FrameSampler instantiation."""
        from capture.camera_stream import CameraStream
        from capture.frame_sampler import FrameSampler

        stream = CameraStream(camera_id="TEST-FS", source=0)
        sampler = FrameSampler(
            camera_stream=stream,
            frame_skip=5,
            resolution=(640, 480),
        )
        self.assertEqual(sampler.frame_number, 0)

    def test_update_skip_rate(self):
        """Test dynamic skip rate adjustment."""
        from capture.camera_stream import CameraStream
        from capture.frame_sampler import FrameSampler

        stream = CameraStream(camera_id="TEST-FS2", source=0)
        sampler = FrameSampler(camera_stream=stream, frame_skip=5)
        sampler.update_skip_rate(10)
        self.assertEqual(sampler._frame_skip, 10)

    def test_skip_rate_minimum(self):
        """Test that skip rate cannot go below 1."""
        from capture.camera_stream import CameraStream
        from capture.frame_sampler import FrameSampler

        stream = CameraStream(camera_id="TEST-MIN", source=0)
        sampler = FrameSampler(camera_stream=stream, frame_skip=5)
        sampler.update_skip_rate(0)
        self.assertEqual(sampler._frame_skip, 1)


class TestStreamManager(unittest.TestCase):
    """Tests for the StreamManager class."""

    def test_stream_manager_creation(self):
        """Test StreamManager with disabled cameras."""
        from capture.stream_manager import StreamManager

        config = {
            "cameras": [
                {
                    "id": "CAM-DISABLED",
                    "source": 0,
                    "enabled": False,
                    "resolution": [640, 480],
                    "fps_target": 15,
                    "frame_skip": 5,
                }
            ]
        }
        sm = StreamManager(config)
        self.assertEqual(len(sm.get_all_camera_ids()), 0)

    def test_stream_manager_repr(self):
        """Test string representation."""
        from capture.stream_manager import StreamManager

        config = {"cameras": []}
        sm = StreamManager(config)
        repr_str = repr(sm)
        self.assertIn("StreamManager", repr_str)
        self.assertIn("total=0", repr_str)

    def test_get_frame_nonexistent_camera(self):
        """Test getting frame from non-existent camera returns None."""
        from capture.stream_manager import StreamManager

        sm = StreamManager({"cameras": []})
        result = sm.get_frame("NONEXISTENT")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
