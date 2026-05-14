"""
SafeWatch — AbuseDetector
Detects sustained/repeated abuse patterns over extended periods.
"""

import time
from collections import defaultdict
from typing import Optional

import numpy as np
from loguru import logger

from detection.person_detector import Person
from detection.pose_estimator import PoseResult
from classifier.skeleton_analyzer import SkeletonAnalyzer
from classifier.velocity_tracker import VelocityTracker
from threats.fight_detector import ThreatEvent


class AbuseDetector:
    """
    Detects sustained abuse patterns — repeated strikes over extended periods
    with consistent dominant/defensive role assignment.
    """

    def __init__(self, config: dict):
        self._config = config.get("threats", {}).get("abuse", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.80)
        self._analyzer = SkeletonAnalyzer()
        self._pair_state: dict[tuple, dict] = defaultdict(lambda: {
            "strike_events": [],
            "dominant_id": None,
            "defensive_id": None,
            "total_frames": 0,
            "alerted": False,
            "last_update": time.time(),
        })
        self._min_tracking_frames = 120
        self._strike_window = 30  # Frames
        logger.info(f"AbuseDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"AbuseDetector(enabled={self._enabled}, tracking={len(self._pair_state)} pairs)"

    def detect(
        self,
        persons: list[Person],
        poses: list[PoseResult],
        velocity_tracker: VelocityTracker,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """Detect abuse scenarios."""
        if not self._enabled or len(persons) < 2:
            return []

        events = []
        pose_map = {p.person_id: p for p in poses}

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                event = self._evaluate_pair(p1, p2, pose_map, velocity_tracker)
                if event is not None:
                    events.append(event)

        # Cleanup stale pairs
        now = time.time()
        stale = [
            k for k, v in self._pair_state.items()
            if now - v["last_update"] > 30
        ]
        for k in stale:
            del self._pair_state[k]

        return events

    def _evaluate_pair(
        self,
        p1: Person,
        p2: Person,
        pose_map: dict[int, PoseResult],
        velocity_tracker: VelocityTracker,
    ) -> Optional[ThreatEvent]:
        """Evaluate abuse pattern between two persons."""
        pair_key = (min(p1.id, p2.id), max(p1.id, p2.id))
        state = self._pair_state[pair_key]
        state["total_frames"] += 1
        state["last_update"] = time.time()

        # Check distance
        dist = np.sqrt(
            (p1.center[0] - p2.center[0])**2 +
            (p1.center[1] - p2.center[1])**2
        )
        if dist > max(p1.width, p2.width) * 3:
            return None

        pose1 = pose_map.get(p1.id)
        pose2 = pose_map.get(p2.id)

        # Determine roles based on wrist velocity asymmetry
        wrist_vel_p1 = max(
            velocity_tracker.get_velocity(p1.id, "left_wrist"),
            velocity_tracker.get_velocity(p1.id, "right_wrist"),
        )
        wrist_vel_p2 = max(
            velocity_tracker.get_velocity(p2.id, "left_wrist"),
            velocity_tracker.get_velocity(p2.id, "right_wrist"),
        )

        # Track strike events
        strike_threshold = 40.0
        if wrist_vel_p1 > strike_threshold or wrist_vel_p2 > strike_threshold:
            striker_id = p1.id if wrist_vel_p1 > wrist_vel_p2 else p2.id
            state["strike_events"].append({
                "frame": state["total_frames"],
                "striker": striker_id,
                "velocity": max(wrist_vel_p1, wrist_vel_p2),
            })

        # Keep only recent strikes
        state["strike_events"] = [
            s for s in state["strike_events"]
            if state["total_frames"] - s["frame"] < 300
        ]

        # Check for consistent role assignment
        if len(state["strike_events"]) >= 3:
            striker_counts: dict[int, int] = {}
            for se in state["strike_events"]:
                sid = se["striker"]
                striker_counts[sid] = striker_counts.get(sid, 0) + 1

            sorted_strikers = sorted(striker_counts.items(), key=lambda x: x[1], reverse=True)
            dominant = sorted_strikers[0]

            if len(sorted_strikers) >= 2:
                dominance_ratio = dominant[1] / (dominant[1] + sorted_strikers[1][1])
            else:
                dominance_ratio = 1.0

            if dominance_ratio > 0.65:
                state["dominant_id"] = dominant[0]
                state["defensive_id"] = p2.id if dominant[0] == p1.id else p1.id

        # Check for helplessness posture in victim
        helplessness_score = 0.0
        if state["defensive_id"] is not None:
            defensive_pose = pose_map.get(state["defensive_id"])
            if defensive_pose is not None:
                orientation = self._analyzer.get_body_orientation(defensive_pose)
                if orientation == "crouching":
                    helplessness_score += 0.3

                lean = self._analyzer.get_body_lean_angle(defensive_pose)
                if lean is not None and lean > 20:
                    helplessness_score += 0.2

                arm_level = self._analyzer.get_arm_raise_level(defensive_pose)
                if arm_level is not None and arm_level < 0.3:
                    helplessness_score += 0.1

        # Calculate abuse confidence
        score = 0.0

        # Sustained pattern (need enough observation time)
        if state["total_frames"] >= self._min_tracking_frames:
            score += 0.2

        # Count strikes per window
        recent_strikes = [
            s for s in state["strike_events"]
            if state["total_frames"] - s["frame"] < self._strike_window
        ]
        strikes_per_window = len(recent_strikes)

        if strikes_per_window >= 3:
            score += 0.3
        elif strikes_per_window >= 2:
            score += 0.15

        # Total strike count
        total_strikes = len(state["strike_events"])
        if total_strikes >= 5:
            score += 0.2
        elif total_strikes >= 3:
            score += 0.1

        # Consistent dominance
        if state["dominant_id"] is not None:
            score += 0.15

        # Victim helplessness
        score += helplessness_score * 0.3

        confidence = min(1.0, score)

        if (confidence >= self._confidence_threshold
                and not state["alerted"]
                and state["total_frames"] >= self._min_tracking_frames):
            state["alerted"] = True

            x1 = min(p1.bbox[0], p2.bbox[0])
            y1 = min(p1.bbox[1], p2.bbox[1])
            x2 = max(p1.bbox[2], p2.bbox[2])
            y2 = max(p1.bbox[3], p2.bbox[3])

            severity = "CRITICAL" if total_strikes >= 5 else "HIGH"

            return ThreatEvent(
                threat_type="ABUSE",
                confidence=round(confidence, 3),
                persons_involved=[p1.id, p2.id],
                location_bbox=(x1, y1, x2, y2),
                description=(
                    f"Sustained abuse pattern detected. Person {state['dominant_id']} "
                    f"repeatedly attacking Person {state['defensive_id']}. "
                    f"Total strikes: {total_strikes}. Duration: {state['total_frames']} frames."
                ),
                severity=severity,
            )

        return None
