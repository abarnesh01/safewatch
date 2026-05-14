"""
SafeWatch Optical Flow
Lucas-Kanade based motion intelligence and divergence analysis.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass

from loguru import logger


@dataclass
class FlowStats:
    """Statistical summary of optical flow in a frame or region."""
    mean_magnitude: float = 0.0
    max_magnitude: float = 0.0
    divergence: float = 0.0
    dominant_direction: float = 0.0  # In degrees
    active_points: int = 0


class OpticalFlowAnalyzer:
    """Analyzes motion patterns using dense and sparse optical flow."""

    def __init__(self, win_size: int = 15, max_level: int = 2,
                 max_corners: int = 200, quality_level: float = 0.01,
                 min_distance: int = 10) -> None:
        self._win_size = win_size
        self._max_level = max_level
        self._lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self._feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._frame_count = 0
        logger.info("OpticalFlowAnalyzer initialized (win_size={})", win_size)

    def analyze(self, frame: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> Optional[FlowStats]:
        """Compute flow statistics between current and previous frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
            return None

        if self._prev_pts is None or len(self._prev_pts) < 10:
            self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
            self._prev_gray = gray
            return None

        try:
            # Calculate sparse optical flow
            curr_pts, status, error = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None, **self._lk_params
            )

            if curr_pts is None:
                return None

            # Filter valid points
            good_new = curr_pts[status == 1]
            good_old = self._prev_pts[status == 1]

            if len(good_new) < 5:
                self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
                self._prev_gray = gray
                return None

            # Calculate vectors
            vectors = good_new - good_old
            magnitudes = np.sqrt(np.sum(vectors**2, axis=1))
            angles = np.arctan2(vectors[:, 1], vectors[:, 0]) * 180 / np.pi

            # Divergence (simplified as standard deviation of direction)
            divergence = float(np.std(angles)) if len(angles) > 1 else 0.0
            
            stats = FlowStats(
                mean_magnitude=float(np.mean(magnitudes)),
                max_magnitude=float(np.max(magnitudes)),
                divergence=divergence,
                dominant_direction=float(np.median(angles)),
                active_points=len(good_new)
            )

            # Update for next frame
            if self._frame_count % 10 == 0:
                # Refresh features periodically
                self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._feature_params)
            else:
                self._prev_pts = good_new.reshape(-1, 1, 2)
            
            self._prev_gray = gray
            self._frame_count += 1
            
            return stats

        except Exception as exc:
            logger.error("Optical flow calculation failed: {}", exc)
            self._prev_gray = None
            return None

    def draw_flow(self, frame: np.ndarray, stats: FlowStats) -> np.ndarray:
        """Visualize flow direction and intensity."""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw motion indicator
        center = (w - 60, 60)
        cv2.circle(annotated, center, 40, (50, 50, 50), -1)
        
        # Arrow indicating dominant direction
        angle_rad = stats.dominant_direction * np.pi / 180
        length = min(stats.mean_magnitude * 5, 35)
        end_pt = (
            int(center[0] + length * np.cos(angle_rad)),
            int(center[1] + length * np.sin(angle_rad))
        )
        
        cv2.arrowedLine(annotated, center, end_pt, (0, 255, 0), 2, tipLength=0.3)
        
        # Text summary
        text = f"Flow: {stats.mean_magnitude:.1f} Div: {stats.divergence:.1f}"
        cv2.putText(annotated, text, (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return annotated

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_pts = None
        self._frame_count = 0
        logger.debug("OpticalFlowAnalyzer reset")
