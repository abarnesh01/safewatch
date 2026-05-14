"""
SafeWatch — HarassmentDetector
Detects harassment scenarios based on sustained proximity, asymmetric body orientation,
and aggressive gesturing patterns.
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


class HarassmentDetector:
    """Detects harassment scenarios through sustained proximity and behavioral analysis."""

    def __init__(self, config: dict):
        self._config = config.get("threats", {}).get("harassment", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.75)
        self._proximity_threshold = self._config.get("proximity_threshold", 0.15)
        self._duration_frames = self._config.get("duration_frames", 60)
        self._analyzer = SkeletonAnalyzer()
        self._pair_tracking: dict[tuple, dict] = defaultdict(lambda: {
            "proximity_frames": 0,
            "first_seen": 0.0,
            "last_seen": 0.0,
            "alerted": False,
            "aggression_score": 0.0,
        })
        logger.info(f"HarassmentDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"HarassmentDetector(enabled={self._enabled}, tracking={len(self._pair_tracking)} pairs)"

    def detect(
        self,
        persons: list[Person],
        poses: list[PoseResult],
        velocity_tracker: VelocityTracker,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """Detect harassment scenarios."""
        if not self._enabled or len(persons) < 2:
            return []

        events = []
        pose_map = {p.person_id: p for p in poses}
        frame_width = 640

        active_pairs = set()

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                pair_key = (min(p1.id, p2.id), max(p1.id, p2.id))
                active_pairs.add(pair_key)

                dist = np.sqrt(
                    (p1.center[0] - p2.center[0])**2 +
                    (p1.center[1] - p2.center[1])**2
                )
                normalized_dist = dist / frame_width

                if normalized_dist < self._proximity_threshold:
                    state = self._pair_tracking[pair_key]
                    state["proximity_frames"] += 1
                    state["last_seen"] = time.time()
                    if state["first_seen"] == 0:
                        state["first_seen"] = time.time()

                    pose1 = pose_map.get(p1.id)
                    pose2 = pose_map.get(p2.id)

                    aggression = self._evaluate_aggression(
                        p1, p2, pose1, pose2, velocity_tracker
                    )
                    state["aggression_score"] = (
                        state["aggression_score"] * 0.8 + aggression * 0.2
                    )

                    if (state["proximity_frames"] >= self._duration_frames
                            and not state["alerted"]):
                        confidence = self._calculate_confidence(state)

                        if confidence >= self._confidence_threshold:
                            state["alerted"] = True
                            x1 = min(p1.bbox[0], p2.bbox[0])
                            y1 = min(p1.bbox[1], p2.bbox[1])
                            x2 = max(p1.bbox[2], p2.bbox[2])
                            y2 = max(p1.bbox[3], p2.bbox[3])

                            severity = "HIGH" if state["aggression_score"] > 0.6 else "MEDIUM"

                            events.append(ThreatEvent(
                                threat_type="HARASSMENT",
                                confidence=round(confidence, 3),
                                persons_involved=[p1.id, p2.id],
                                location_bbox=(x1, y1, x2, y2),
                                description=(
                                    f"Potential harassment detected. Sustained close proximity "
                                    f"({state['proximity_frames']} frames) with aggressive behavior patterns."
                                ),
                                severity=severity,
                            ))
                else:
                    if pair_key in self._pair_tracking:
                        state = self._pair_tracking[pair_key]
                        state["proximity_frames"] = max(0, state["proximity_frames"] - 2)

        stale = [k for k in self._pair_tracking if k not in active_pairs]
        for k in stale:
            if time.time() - self._pair_tracking[k]["last_seen"] > 10:
                del self._pair_tracking[k]

        return events

    def _evaluate_aggression(
        self,
        p1: Person,
        p2: Person,
        pose1: Optional[PoseResult],
        pose2: Optional[PoseResult],
        velocity_tracker: VelocityTracker,
    ) -> float:
        """Score aggression level between two persons."""
        score = 0.0

        # Check if one person is stationary and the other approaching
        vel1 = velocity_tracker.get_average_velocity(p1.id)
        vel2 = velocity_tracker.get_average_velocity(p2.id)

        if (vel1 > 10 and vel2 < 3) or (vel2 > 10 and vel1 < 3):
            score += 0.3

        # Check asymmetric body orientation
        if pose1 is not None and pose2 is not None:
            dir1 = self._analyzer.get_facing_direction(pose1)
            dir2 = self._analyzer.get_facing_direction(pose2)

            if dir1 is not None and dir2 is not None:
                if dir1 == "forward" and dir2 != "forward":
                    score += 0.2
                elif dir2 == "forward" and dir1 != "forward":
                    score += 0.2

        # Check arm raises (aggressive gesturing)
        if pose1 is not None:
            arm1 = self._analyzer.get_arm_raise_level(pose1)
            if arm1 is not None and arm1 > 0.5:
                score += 0.2

        if pose2 is not None:
            arm2 = self._analyzer.get_arm_raise_level(pose2)
            if arm2 is not None and arm2 > 0.5:
                score += 0.2

        # Check if one person is backed against area (cornered)
        frame_margin = 0.1
        for person in [p1, p2]:
            nx = person.center[0] / 640
            ny = person.center[1] / 480
            if nx < frame_margin or nx > (1 - frame_margin) or ny > (1 - frame_margin):
                score += 0.1

        return min(1.0, score)

    def _calculate_confidence(self, state: dict) -> float:
        """Calculate overall harassment confidence."""
        duration_factor = min(1.0, state["proximity_frames"] / (self._duration_frames * 1.5))
        aggression_factor = state["aggression_score"]

        confidence = duration_factor * 0.6 + aggression_factor * 0.4
        return min(1.0, confidence)
