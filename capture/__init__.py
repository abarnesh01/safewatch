"""
SafeWatch Capture Package
Provides camera stream management and frame sampling.
"""

from capture.camera_stream import CameraStream
from capture.frame_sampler import FrameSampler
from capture.stream_manager import StreamManager

__all__ = ["CameraStream", "FrameSampler", "StreamManager"]
