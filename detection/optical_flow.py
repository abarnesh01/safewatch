"""
SafeWatch — OpticalFlowAnalyzer
Lucas-Kanade optical flow for motion analysis and crowd divergence detection.
"""

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from loguru import logger


@dataclass
class FlowResult:
    """Contains optical flow analysis results."""
    mean_magnitude: float
    max_magnitude: float
    flow_vectors: list[tuple]
    divergence_score: float
    motion_regions: list[tuple]

    def __repr__(self) -> str:
        return (
            f"FlowResult(mean_mag={self.mean_magnitude:.2f}, "
            f"max_mag={self.max_magnitude:.2f}, "
            f"divergence={self.divergence_score:.2f}, "
            f"regions={len(self.motion_regions)})"
        )


class OpticalFlowAnalyzer:
    """
    Lucas-Kanade optical flow analyzer for detecting motion patterns,
    sudden movements, and crowd divergence.
    """

    def __init__(self, config: dict):
        self._config = config.get("detection", {})
        self._enabled = self._config.get("enable_optical_flow", True)
        self._lock = threading.Lock()

        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self._feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None
        self._frame_count = 0
        self._reset_interval = 30
        self._sudden_motion_threshold = 15.0

        logger.info(f"OpticalFlowAnalyzer initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        has_prev = self._prev_gray is not None
        return (
            f"OpticalFlowAnalyzer(enabled={self._enabled}, "
            f"has_previous={has_prev}, frame_count={self._frame_count})"
        )

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect good features to track in the grayscale frame."""
        points = cv2.goodFeaturesToTrack(gray, **self._feature_params)
        return points

    def analyze(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> FlowResult:
        """
        Analyze optical flow between two consecutive frames.

        Args:
            prev_frame: Previous BGR frame
            curr_frame: Current BGR frame

        Returns:
            FlowResult with motion analysis data
        """
        if not self._enabled:
            return FlowResult(
                mean_magnitude=0.0,
                max_magnitude=0.0,
                flow_vectors=[],
                divergence_score=0.0,
                motion_regions=[],
            )

        with self._lock:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            self._frame_count += 1

            if self._prev_points is None or self._frame_count % self._reset_interval == 0:
                self._prev_points = self._detect_features(prev_gray)
                self._prev_gray = prev_gray

            if self._prev_points is None or len(self._prev_points) < 5:
                self._prev_points = self._detect_features(prev_gray)
                self._prev_gray = prev_gray
                if self._prev_points is None or len(self._prev_points) < 5:
                    return FlowResult(
                        mean_magnitude=0.0,
                        max_magnitude=0.0,
                        flow_vectors=[],
                        divergence_score=0.0,
                        motion_regions=[],
                    )

            try:
                next_points, status, errors = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, curr_gray, self._prev_points, None, **self._lk_params
                )
            except Exception as e:
                logger.error(f"Optical flow calculation error: {e}")
                self._prev_gray = curr_gray
                self._prev_points = self._detect_features(curr_gray)
                return FlowResult(
                    mean_magnitude=0.0,
                    max_magnitude=0.0,
                    flow_vectors=[],
                    divergence_score=0.0,
                    motion_regions=[],
                )

            if next_points is None or status is None:
                self._prev_gray = curr_gray
                self._prev_points = self._detect_features(curr_gray)
                return FlowResult(
                    mean_magnitude=0.0,
                    max_magnitude=0.0,
                    flow_vectors=[],
                    divergence_score=0.0,
                    motion_regions=[],
                )

            good_mask = status.flatten() == 1
            good_prev = self._prev_points[good_mask]
            good_next = next_points[good_mask]

            if len(good_prev) == 0:
                self._prev_gray = curr_gray
                self._prev_points = self._detect_features(curr_gray)
                return FlowResult(
                    mean_magnitude=0.0,
                    max_magnitude=0.0,
                    flow_vectors=[],
                    divergence_score=0.0,
                    motion_regions=[],
                )

            displacements = good_next - good_prev
            magnitudes = np.sqrt(displacements[:, 0, 0]**2 + displacements[:, 0, 1]**2)

            mean_mag = float(np.mean(magnitudes))
            max_mag = float(np.max(magnitudes))

            flow_vectors = []
            for i in range(len(good_prev)):
                px, py = good_prev[i].ravel()
                nx, ny = good_next[i].ravel()
                flow_vectors.append((float(px), float(py), float(nx), float(ny), float(magnitudes[i])))

            divergence_score = self._calculate_divergence(good_prev, good_next, displacements)

            motion_regions = self._find_motion_regions(
                curr_gray, displacements, good_next, magnitudes
            )

            self._prev_gray = curr_gray
            self._prev_points = good_next.reshape(-1, 1, 2)

            return FlowResult(
                mean_magnitude=mean_mag,
                max_magnitude=max_mag,
                flow_vectors=flow_vectors,
                divergence_score=divergence_score,
                motion_regions=motion_regions,
            )

    def _calculate_divergence(
        self,
        prev_pts: np.ndarray,
        next_pts: np.ndarray,
        displacements: np.ndarray,
    ) -> float:
        """
        Calculate flow divergence — high values indicate people moving in all directions
        (panic-like behavior).
        """
        if len(displacements) < 3:
            return 0.0

        center = np.mean(prev_pts, axis=0).reshape(1, 2)
        angles = np.arctan2(displacements[:, 0, 1], displacements[:, 0, 0])

        angle_std = float(np.std(angles))
        magnitudes = np.sqrt(displacements[:, 0, 0]**2 + displacements[:, 0, 1]**2)
        mean_mag = float(np.mean(magnitudes))

        divergence = angle_std * mean_mag
        return divergence

    def _find_motion_regions(
        self,
        gray: np.ndarray,
        displacements: np.ndarray,
        points: np.ndarray,
        magnitudes: np.ndarray,
    ) -> list[tuple]:
        """Find rectangular regions with significant motion."""
        regions = []
        threshold = 5.0

        high_motion_mask = magnitudes > threshold
        high_motion_pts = points[high_motion_mask]

        if len(high_motion_pts) < 2:
            return regions

        for i in range(0, len(high_motion_pts), 5):
            chunk = high_motion_pts[i:i+5]
            if len(chunk) < 2:
                continue
            xs = chunk[:, 0, 0]
            ys = chunk[:, 0, 1]
            x1 = int(np.min(xs)) - 10
            y1 = int(np.min(ys)) - 10
            x2 = int(np.max(xs)) + 10
            y2 = int(np.max(ys)) + 10
            regions.append((x1, y1, x2, y2))

        return regions

    def detect_sudden_motion(self, flow_result: FlowResult) -> tuple[bool, float]:
        """
        Detect if there is a sudden spike in motion.

        Args:
            flow_result: FlowResult from analyze()

        Returns:
            Tuple of (is_sudden, magnitude)
        """
        is_sudden = flow_result.max_magnitude > self._sudden_motion_threshold
        return is_sudden, flow_result.max_magnitude

    def detect_crowd_divergence(self, flow_result: FlowResult) -> tuple[bool, float]:
        """
        Detect crowd divergence pattern (people running in all directions).

        Args:
            flow_result: FlowResult from analyze()

        Returns:
            Tuple of (is_divergent, divergence_score)
        """
        threshold = self._config.get("crowd_panic", {}).get("flow_divergence_threshold", 8.0)
        is_divergent = flow_result.divergence_score > threshold
        return is_divergent, flow_result.divergence_score

    def reset(self):
        """Reset the analyzer state."""
        with self._lock:
            self._prev_gray = None
            self._prev_points = None
            self._frame_count = 0
            logger.debug("OpticalFlowAnalyzer reset")
