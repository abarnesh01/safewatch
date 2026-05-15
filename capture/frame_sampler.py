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
        config: Optional[dict] = None,
    ):
        self._stream = camera_stream
        self._frame_skip = frame_skip
        self._resolution = resolution
        self._base_threshold = motion_threshold
        self._current_threshold = motion_threshold
        
        # Adaptive parameters
        sampler_cfg = config.get("sampling", {}) if config else {}
        self._adaptive_enabled = sampler_cfg.get("adaptive_motion", True)
        self._sensitivity = sampler_cfg.get("sensitivity", 1.0)
        
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=sampler_cfg.get("bg_var_threshold", 50),
            detectShadows=False,
        )
        
        self._motion_history = deque(maxlen=100)
        self._light_level_history = deque(maxlen=50)
        self._frame_number = 0
        self._last_processed_time = 0.0
        
        logger.info(
            f"FrameSampler initialized for {camera_stream.camera_id}: "
            f"adaptive={self._adaptive_enabled}, base_threshold={motion_threshold}"
        )

    def __repr__(self) -> str:
        return (
            f"FrameSampler(camera={self._stream.camera_id}, "
            f"threshold={self._current_threshold:.1f}, skip={self._frame_skip})"
        )

    def _detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect motion with adaptive thresholding and low-light compensation.
        """
        # 1. Pre-process for noise suppression
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Track lighting levels for compensation
        avg_brightness = np.mean(blurred)
        self._light_level_history.append(avg_brightness)
        
        # Low-light compensation factor
        light_factor = 1.0
        if len(self._light_level_history) > 10:
            avg_light = np.mean(self._light_level_history)
            if avg_light < 50: # Dim scene
                light_factor = 1.5 # Increase sensitivity in low light
            elif avg_light > 200: # Very bright scene
                light_factor = 0.8 # Decrease sensitivity to avoid glare triggers

        # 3. Apply background subtraction
        fg_mask = self._bg_subtractor.apply(blurred)
        
        # Noise suppression (remove small flickers)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        motion_pixels = cv2.countNonZero(fg_mask)
        self._motion_history.append(motion_pixels)
        
        # 4. Adaptive threshold adjustment
        if self._adaptive_enabled and len(self._motion_history) > 20:
            # Baseline is the median noise level in the scene
            baseline_noise = np.median(self._motion_history)
            # Threshold scales with baseline noise + light factor
            self._current_threshold = (self._base_threshold + baseline_noise * 0.5) / (self._sensitivity * light_factor)
        else:
            self._current_threshold = self._base_threshold / (self._sensitivity * light_factor)

        # 5. Final trigger decision
        is_motion = motion_pixels > self._current_threshold
        return is_motion

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
