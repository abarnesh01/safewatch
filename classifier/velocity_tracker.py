"""
SafeWatch — VelocityTracker
Tracks position history and computes velocities/accelerations for each person.
"""

import time
import threading
from collections import defaultdict, deque
from typing import Optional

import numpy as np
from loguru import logger

from detection.pose_estimator import PoseResult


class VelocityTracker:
    """
    Tracks per-person joint position history and computes velocities,
    accelerations, and trajectories over time.
    """

    def __init__(self, max_history: int = 60, cleanup_timeout: float = 5.0):
        self._max_history = max_history
        self._cleanup_timeout = cleanup_timeout
        self._lock = threading.Lock()
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._last_seen: dict[int, float] = {}
        logger.info(f"VelocityTracker initialized (history={max_history}, timeout={cleanup_timeout}s)")

    def __repr__(self) -> str:
        with self._lock:
            return f"VelocityTracker(tracking={len(self._history)} persons)"

    def update(self, person_id: int, pose_result: PoseResult, timestamp: float):
        """
        Record a new pose observation for a person.

        Args:
            person_id: The tracked person's ID
            pose_result: Current PoseResult
            timestamp: Unix timestamp of the observation
        """
        with self._lock:
            entry = {
                "timestamp": timestamp,
                "keypoints": {},
            }

            for name in ["nose", "left_shoulder", "right_shoulder",
                         "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                         "left_hip", "right_hip", "left_knee", "right_knee",
                         "left_ankle", "right_ankle"]:
                kp = pose_result.get_landmark(name)
                if kp is not None:
                    entry["keypoints"][name] = {
                        "x": kp["x"],
                        "y": kp["y"],
                        "abs_x": kp.get("abs_x", kp["x"]),
                        "abs_y": kp.get("abs_y", kp["y"]),
                    }

            self._history[person_id].append(entry)
            self._last_seen[person_id] = timestamp

        self._cleanup()

    def _cleanup(self):
        """Remove persons not seen for cleanup_timeout seconds."""
        now = time.time()
        to_remove = []
        with self._lock:
            for pid, last_time in self._last_seen.items():
                if now - last_time > self._cleanup_timeout:
                    to_remove.append(pid)

            for pid in to_remove:
                if pid in self._history:
                    del self._history[pid]
                if pid in self._last_seen:
                    del self._last_seen[pid]

    def get_velocity(self, person_id: int, joint_name: str, window: int = 1) -> float:
        """
        Get the velocity of a joint in pixels per second.

        Args:
            person_id: Person ID to query
            joint_name: Name of the joint
            window: Number of recent samples to average over (1 = latest only)

        Returns:
            Velocity in pixels per second, or 0.0 if unknown.
        """
        with self._lock:
            history = self._history.get(person_id)
            if history is None or len(history) < 2:
                return 0.0

            if window <= 1:
                curr = history[-1]
                prev = history[-2]
                return self._compute_instant_velocity(curr, prev, joint_name)
            
            # Compute average velocity over a window
            velocities = []
            samples = list(history)[-window-1:]
            for i in range(1, len(samples)):
                v = self._compute_instant_velocity(samples[i], samples[i-1], joint_name)
                if v > 0:
                    velocities.append(v)
            
            return float(np.mean(velocities)) if velocities else 0.0

    def _compute_instant_velocity(self, curr: dict, prev: dict, joint_name: str) -> float:
        """Helper to compute velocity between two frames."""
        curr_kp = curr["keypoints"].get(joint_name)
        prev_kp = prev["keypoints"].get(joint_name)

        if curr_kp is None or prev_kp is None:
            return 0.0

        dt = curr["timestamp"] - prev["timestamp"]
        if dt <= 0:
            return 0.0

        dx = curr_kp["abs_x"] - prev_kp["abs_x"]
        dy = curr_kp["abs_y"] - prev_kp["abs_y"]
        distance = np.sqrt(dx**2 + dy**2)

        return float(distance / dt)

    def get_acceleration(self, person_id: int, joint_name: str) -> float:
        """
        Get the acceleration of a joint in pixels per second squared.

        Args:
            person_id: Person ID
            joint_name: Joint name

        Returns:
            Acceleration in pixels/s², or 0.0 if unknown.
        """
        with self._lock:
            history = self._history.get(person_id)
            if history is None or len(history) < 3:
                return 0.0

            entries = [history[-3], history[-2], history[-1]]

        velocities = []
        for i in range(1, len(entries)):
            curr_kp = entries[i]["keypoints"].get(joint_name)
            prev_kp = entries[i-1]["keypoints"].get(joint_name)

            if curr_kp is None or prev_kp is None:
                return 0.0

            dt = entries[i]["timestamp"] - entries[i-1]["timestamp"]
            if dt <= 0:
                return 0.0

            dx = curr_kp["abs_x"] - prev_kp["abs_x"]
            dy = curr_kp["abs_y"] - prev_kp["abs_y"]
            vel = np.sqrt(dx**2 + dy**2) / dt
            velocities.append((vel, entries[i]["timestamp"]))

        if len(velocities) < 2:
            return 0.0

        dv = velocities[1][0] - velocities[0][0]
        dt = velocities[1][1] - velocities[0][1]

        if dt <= 0:
            return 0.0

        return float(dv / dt)

    def get_trajectory(self, person_id: int, n_frames: int = 10) -> list[tuple[float, float]]:
        """
        Get recent position trajectory for a person (using hip center).

        Args:
            person_id: Person ID
            n_frames: Number of recent frames to include

        Returns:
            List of (x, y) positions.
        """
        with self._lock:
            history = self._history.get(person_id)
            if history is None:
                return []

            positions = []
            entries = list(history)[-n_frames:]
            for entry in entries:
                lh = entry["keypoints"].get("left_hip")
                rh = entry["keypoints"].get("right_hip")
                if lh is not None and rh is not None:
                    cx = (lh["abs_x"] + rh["abs_x"]) / 2
                    cy = (lh["abs_y"] + rh["abs_y"]) / 2
                    positions.append((float(cx), float(cy)))
                elif lh is not None:
                    positions.append((float(lh["abs_x"]), float(lh["abs_y"])))
                elif rh is not None:
                    positions.append((float(rh["abs_x"]), float(rh["abs_y"])))

            return positions

    def get_relative_velocity(self, person_id_1: int, person_id_2: int) -> float:
        """
        Get closing/opening speed between two persons in pixels per second.
        Positive = closing, Negative = moving apart.

        Args:
            person_id_1: First person ID
            person_id_2: Second person ID

        Returns:
            Relative velocity (positive = approaching), or 0.0 if unknown.
        """
        with self._lock:
            h1 = self._history.get(person_id_1)
            h2 = self._history.get(person_id_2)

            if h1 is None or h2 is None or len(h1) < 2 or len(h2) < 2:
                return 0.0

        def _get_center(entry: dict) -> Optional[tuple]:
            lh = entry["keypoints"].get("left_hip")
            rh = entry["keypoints"].get("right_hip")
            if lh and rh:
                return ((lh["abs_x"] + rh["abs_x"]) / 2, (lh["abs_y"] + rh["abs_y"]) / 2)
            return None

        with self._lock:
            curr1 = _get_center(h1[-1])
            prev1 = _get_center(h1[-2])
            curr2 = _get_center(h2[-1])
            prev2 = _get_center(h2[-2])

            dt1 = h1[-1]["timestamp"] - h1[-2]["timestamp"]
            dt2 = h2[-1]["timestamp"] - h2[-2]["timestamp"]

        if any(c is None for c in [curr1, prev1, curr2, prev2]):
            return 0.0

        dt = (dt1 + dt2) / 2
        if dt <= 0:
            return 0.0

        prev_dist = np.sqrt((prev1[0] - prev2[0])**2 + (prev1[1] - prev2[1])**2)
        curr_dist = np.sqrt((curr1[0] - curr2[0])**2 + (curr1[1] - curr2[1])**2)

        closing_speed = (prev_dist - curr_dist) / dt
        return float(closing_speed)

    def get_average_velocity(self, person_id: int, n_frames: int = 5) -> float:
        """
        Get the average overall velocity over the last N frames.

        Returns:
            Average velocity in pixels/s, or 0.0.
        """
        with self._lock:
            history = self._history.get(person_id)
            if history is None or len(history) < 2:
                return 0.0

            entries = list(history)[-n_frames:]

        velocities = []
        for joint in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]:
            for i in range(1, len(entries)):
                curr_kp = entries[i]["keypoints"].get(joint)
                prev_kp = entries[i-1]["keypoints"].get(joint)
                if curr_kp is None or prev_kp is None:
                    continue
                dt = entries[i]["timestamp"] - entries[i-1]["timestamp"]
                if dt <= 0:
                    continue
                dx = curr_kp["abs_x"] - prev_kp["abs_x"]
                dy = curr_kp["abs_y"] - prev_kp["abs_y"]
                vel = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(vel)

        if not velocities:
            return 0.0
        return float(np.mean(velocities))

    def get_history_length(self, person_id: int) -> int:
        """Get the number of stored frames for a person."""
        with self._lock:
            history = self._history.get(person_id)
            return len(history) if history else 0

    def get_all_tracked_ids(self) -> list[int]:
        """Get list of all currently tracked person IDs."""
        with self._lock:
            return list(self._history.keys())

    def clear(self):
        """Clear all tracking data."""
        with self._lock:
            self._history.clear()
            self._last_seen.clear()
            logger.debug("VelocityTracker cleared")
