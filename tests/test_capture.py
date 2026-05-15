"""
SafeWatch — Comprehensive Capture Tests
Unit tests for CameraStream and StreamManager including life-cycle and registration.
"""

import unittest
import time
from capture.camera_stream import CameraStream
from capture.stream_manager import StreamManager

class TestSafeWatchCapture(unittest.TestCase):
    """Tests for the camera capture and stream management system."""

    def test_camera_stream_lifecycle(self):
        """Test CameraStream start/stop cycle."""
        # Using a source that likely won't open but thread should still start
        cam = CameraStream(camera_id="test_cam", source=99, fps_target=10)
        self.assertEqual(cam.camera_id, "test_cam")
        
        cam.start()
        self.assertTrue(cam.is_running())
        
        status = cam.get_status()
        self.assertEqual(status["camera_id"], "test_cam")
        self.assertEqual(status["running"], True)
        
        cam.stop()
        self.assertFalse(cam.is_running())

    def test_stream_manager_orchestration(self):
        """Test StreamManager's ability to manage multiple cameras."""
        manager = StreamManager()
        manager.add_camera("cam1", source=0, camera_name="Front")
        manager.add_camera("cam2", source=1, camera_name="Back")
        
        self.assertEqual(manager.get_camera_count(), 2)
        ids = manager.get_camera_ids()
        self.assertIn("cam1", ids)
        self.assertIn("cam2", ids)
        
        # Test individual status
        status1 = manager.get_camera_status("cam1")
        self.assertEqual(status1["name"], "Front")
        
        # Cleanup
        manager.stop_all()

    def test_camera_stream_fps_tracking(self):
        """Test that FPS tracking variables are initialized correctly."""
        cam = CameraStream(camera_id="fps_test", source=0)
        self.assertEqual(cam.get_fps(), 0.0)

if __name__ == "__main__":
    unittest.main()
