"""
SafeWatch Frame Sampler
Motion-aware smart frame sampling with background subtraction.
"""

import cv2
import numpy as np
from loguru import logger


class FrameSampler:
    """Intelligent frame sampling based on motion detection."""

    def __init__(self, min_motion_threshold: int = 500,
                 bg_history: int = 500, bg_threshold: int = 50,
                 skip_no_motion: int = 5, skip_motion: int = 1) -> None:
        self._min_motion = min_motion_threshold
        self._skip_no_motion = skip_no_motion
        self._skip_motion = skip_motion
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=bg_history,
            varThreshold=bg_threshold,
            detectShadows=False,
        )
        self._frame_counter = 0
        self._last_motion_score = 0.0
        self._motion_detected = False
        logger.info("FrameSampler initialized (motion_thresh={})", min_motion_threshold)

    def should_process(self, frame: np.ndarray) -> bool:
        """Determine if the frame should be processed based on motion."""
        self._frame_counter += 1
        motion_score = self._compute_motion(frame)
        self._last_motion_score = motion_score
        self._motion_detected = motion_score > self._min_motion

        if self._motion_detected:
            skip = self._skip_motion
        else:
            skip = self._skip_no_motion

        return (self._frame_counter % max(skip, 1)) == 0

    def _compute_motion(self, frame: np.ndarray) -> float:
        """Compute motion score using background subtraction."""
        try:
            small = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mask = self._bg_subtractor.apply(gray)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            motion_pixels = cv2.countNonZero(mask)
            return float(motion_pixels)
        except Exception as exc:
            logger.error("Motion computation error: {}", exc)
            return 0.0

    @property
    def motion_score(self) -> float:
        return self._last_motion_score

    @property
    def motion_detected(self) -> bool:
        return self._motion_detected

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    def reset(self) -> None:
        self._frame_counter = 0
        self._last_motion_score = 0.0
        self._motion_detected = False
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False,
        )
        logger.debug("FrameSampler reset")
