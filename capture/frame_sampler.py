"""
SafeWatch — FrameSampler
Smart frame sampling with motion detection and adaptive skip rates.
"""

import time
from typing import Optional, Generator

import cv2
import numpy as np
from loguru import logger

from capture.camera_stream import CameraStream


class FrameSampler:
    """
    Smart frame sampler that applies skip-based and motion-based sampling
    to reduce processing load while ensuring critical frames are captured.
    """

    def __init__(
        self,
        camera_stream: CameraStream,
        frame_skip: int = 5,
        resolution: tuple[int, int] = (640, 480),
        motion_threshold: float = 500.0,
    ):
        self._stream = camera_stream
        self._frame_skip = frame_skip
        self._resolution = resolution
        self._motion_threshold = motion_threshold
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False,
        )
        self._frame_number = 0
        self._last_processed_time = 0.0
        logger.info(
            f"FrameSampler created for {camera_stream.camera_id}: "
            f"skip={frame_skip}, resolution={resolution}"
        )

    def __repr__(self) -> str:
        return (
            f"FrameSampler(camera={self._stream.camera_id}, "
            f"skip={self._frame_skip}, frame_num={self._frame_number})"
        )

    def _detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect if there is significant motion in the frame using background subtraction.

        Args:
            frame: Input BGR frame

        Returns:
            True if significant motion detected
        """
        small = cv2.resize(frame, (160, 120))
        fg_mask = self._bg_subtractor.apply(small)
        motion_pixels = cv2.countNonZero(fg_mask)
        return motion_pixels > self._motion_threshold

    def get_frame(self) -> Generator[dict, None, None]:
        """
        Generator that yields processed frames with metadata.

        Yields:
            Dict with keys:
                - frame: np.ndarray (BGR image at configured resolution)
                - camera_id: str
                - timestamp: float (unix timestamp)
                - frame_number: int
                - has_motion: bool
        """
        while self._stream.is_running():
            raw_frame = self._stream.read()

            if raw_frame is None:
                time.sleep(0.01)
                continue

            self._frame_number += 1
            has_motion = self._detect_motion(raw_frame)

            if self._frame_number % self._frame_skip != 0 and not has_motion:
                continue

            if raw_frame.shape[1] != self._resolution[0] or raw_frame.shape[0] != self._resolution[1]:
                frame = cv2.resize(raw_frame, self._resolution)
            else:
                frame = raw_frame

            now = time.time()
            self._last_processed_time = now

            yield {
                "frame": frame,
                "camera_id": self._stream.camera_id,
                "timestamp": now,
                "frame_number": self._frame_number,
                "has_motion": has_motion,
            }

    def update_skip_rate(self, n: int):
        """
        Dynamically adjust the frame skip rate.

        Args:
            n: New frame skip value (process every Nth frame)
        """
        old = self._frame_skip
        self._frame_skip = max(1, n)
        logger.info(
            f"[{self._stream.camera_id}] Frame skip updated: {old} → {self._frame_skip}"
        )

    def reset_background(self):
        """Reset the background subtractor model."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False,
        )
        logger.debug(f"[{self._stream.camera_id}] Background model reset")

    @property
    def frame_number(self) -> int:
        return self._frame_number
