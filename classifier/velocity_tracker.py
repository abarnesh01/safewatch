"""
SafeWatch Velocity Tracker
Tracks person and joint movement speed over time for threat analysis.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from loguru import logger
from detection.pose_estimator import PersonPose


class VelocityTracker:
    """Tracks velocities of individuals and their body parts across frames."""

    def __init__(self, window_size: int = 10) -> None:
        self._window_size = window_size
        # Map of person_id -> joint_name -> deque of (timestamp, x, y)
        self._history: Dict[int, Dict[str, deque]] = {}
        logger.info("VelocityTracker initialized (window_size={})", window_size)

    def track(self, pose: PersonPose) -> Dict[str, float]:
        """Update history and compute current velocities for a person's joints."""
        pid = pose.person_id
        now = time.time()
        velocities = {}

        if pid not in self._history:
            self._history[pid] = {}

        for name, lm in pose.landmarks.items():
            if name not in self._history[pid]:
                self._history[pid][name] = deque(maxlen=self._window_size)
            
            # Store current position
            self._history[pid][name].append((now, lm.x, lm.y))
            
            # Calculate velocity if enough history
            if len(self._history[pid][name]) >= 2:
                hist = self._history[pid][name]
                dt = hist[-1][0] - hist[0][0]
                
                if dt > 0:
                    dx = hist[-1][1] - hist[0][1]
                    dy = hist[-1][2] - hist[0][2]
                    dist = np.sqrt(dx**2 + dy**2)
                    velocities[f"{name}_velocity"] = dist / dt
                else:
                    velocities[f"{name}_velocity"] = 0.0
            else:
                velocities[f"{name}_velocity"] = 0.0

        # Overall person velocity (center of mass/hip)
        hip_keys = ["left_hip", "right_hip"]
        if all(k in velocities for k in [f"{h}_velocity" for h in hip_keys]):
            velocities["person_velocity"] = (velocities["left_hip_velocity"] + 
                                             velocities["right_hip_velocity"]) / 2
        else:
            # Fallback to mean of all available joint velocities
            if velocities:
                velocities["person_velocity"] = sum(velocities.values()) / len(velocities)
            else:
                velocities["person_velocity"] = 0.0

        return velocities

    def cleanup(self, active_person_ids: List[int]) -> None:
        """Remove history for persons no longer in view."""
        pids_to_remove = [pid for pid in self._history if pid not in active_person_ids]
        for pid in pids_to_remove:
            del self._history[pid]
            logger.debug("Cleaned up velocity history for person {}", pid)

    def reset(self) -> None:
        self._history.clear()
        logger.debug("VelocityTracker reset")
