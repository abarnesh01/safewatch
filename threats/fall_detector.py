"""
SafeWatch — FallDetector
Detects falls using hip drop velocity, horizontal body detection, and stillness tracking.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from detection.person_detector import Person
from detection.pose_estimator import PoseResult
from classifier.skeleton_analyzer import SkeletonAnalyzer
from classifier.velocity_tracker import VelocityTracker
from threats.fight_detector import ThreatEvent


class FallDetector:
    """
    Detects falls using a state machine: STANDING → FALLING → FALLEN → STATIONARY_FALLEN.
    Uses hip drop velocity, horizontal body angle, and post-fall stillness.
    """

    def __init__(self, config: dict):
        self._config = config.get("threats", {}).get("fall", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.78)
        self._hip_drop_threshold = self._config.get("hip_drop_threshold", 80)
        self._stillness_frames = self._config.get("stillness_frames", 30)
        self._analyzer = SkeletonAnalyzer()
        self._person_states: dict[int, dict] = defaultdict(lambda: {
            "state": "STANDING",
            "fall_start_time": 0.0,
            "stillness_counter": 0,
            "last_hip_y": None,
            "hip_history": [],
            "alerted": False,
        })
        logger.info(f"FallDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"FallDetector(enabled={self._enabled}, tracking={len(self._person_states)} persons)"

    def detect(
        self,
        persons: list[Person],
        poses: list[PoseResult],
        velocity_tracker: VelocityTracker,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """Detect fall scenarios."""
        if not self._enabled:
            return []

        events = []
        pose_map = {p.person_id: p for p in poses}

        for person in persons:
            pose = pose_map.get(person.id)
            if pose is None:
                continue

            event = self._evaluate_person(person, pose, velocity_tracker)
            if event is not None:
                events.append(event)

        tracked_ids = {p.id for p in persons}
        stale = [pid for pid in self._person_states if pid not in tracked_ids]
        for pid in stale:
            state = self._person_states[pid]
            if time.time() - state.get("fall_start_time", 0) > 30:
                del self._person_states[pid]

        return events

    def _evaluate_person(
        self,
        person: Person,
        pose: PoseResult,
        velocity_tracker: VelocityTracker,
    ) -> Optional[ThreatEvent]:
        """Evaluate fall state machine for a single person."""
        state = self._person_states[person.id]
        is_horizontal = self._analyzer.is_person_horizontal(pose)

        left_hip = pose.get_landmark("left_hip")
        right_hip = pose.get_landmark("right_hip")
        current_hip_y = None
        if left_hip and right_hip:
            current_hip_y = (left_hip["y"] + right_hip["y"]) / 2

        hip_drop_velocity = 0.0
        if current_hip_y is not None:
            state["hip_history"].append(current_hip_y)
            if len(state["hip_history"]) > 60:
                state["hip_history"] = state["hip_history"][-60:]

            if state["last_hip_y"] is not None:
                hip_drop_velocity = (current_hip_y - state["last_hip_y"]) * 1000
            state["last_hip_y"] = current_hip_y

        avg_vel = velocity_tracker.get_average_velocity(person.id)
        is_still = avg_vel < 3.0

        current_state = state["state"]

        if current_state == "STANDING":
            if hip_drop_velocity > self._hip_drop_threshold:
                state["state"] = "FALLING"
                state["fall_start_time"] = time.time()
                state["stillness_counter"] = 0
                state["alerted"] = False
                logger.debug(f"Person {person.id}: STANDING → FALLING (hip_drop={hip_drop_velocity:.1f})")

        elif current_state == "FALLING":
            if is_horizontal is True:
                state["state"] = "FALLEN"
                state["stillness_counter"] = 0
                logger.debug(f"Person {person.id}: FALLING → FALLEN")
            elif time.time() - state["fall_start_time"] > 3.0:
                orientation = self._analyzer.get_body_orientation(pose)
                if orientation == "standing":
                    state["state"] = "STANDING"
                else:
                    state["state"] = "FALLEN"

        elif current_state == "FALLEN":
            if is_still:
                state["stillness_counter"] += 1
            else:
                state["stillness_counter"] = max(0, state["stillness_counter"] - 2)

            if state["stillness_counter"] >= self._stillness_frames:
                state["state"] = "STATIONARY_FALLEN"
                logger.debug(f"Person {person.id}: FALLEN → STATIONARY_FALLEN")

            orientation = self._analyzer.get_body_orientation(pose)
            if orientation == "standing":
                state["state"] = "STANDING"
                state["stillness_counter"] = 0
                return None

            if not state["alerted"]:
                state["alerted"] = True
                confidence = self._calculate_fall_confidence(
                    is_horizontal, hip_drop_velocity, is_still
                )
                if confidence >= self._confidence_threshold:
                    return ThreatEvent(
                        threat_type="FALL",
                        confidence=round(confidence, 3),
                        persons_involved=[person.id],
                        location_bbox=person.bbox,
                        description=f"Person fell down. Hip drop velocity: {hip_drop_velocity:.1f}",
                        severity="MEDIUM",
                    )

        elif current_state == "STATIONARY_FALLEN":
            if is_still:
                state["stillness_counter"] += 1
            else:
                state["stillness_counter"] = max(0, state["stillness_counter"] - 5)

            orientation = self._analyzer.get_body_orientation(pose)
            if orientation == "standing":
                state["state"] = "STANDING"
                state["stillness_counter"] = 0
                return None

            if state["stillness_counter"] >= self._stillness_frames * 2 and not state.get("critical_alerted"):
                state["critical_alerted"] = True
                return ThreatEvent(
                    threat_type="FALL",
                    confidence=0.92,
                    persons_involved=[person.id],
                    location_bbox=person.bbox,
                    description="Person has been on the ground motionless for extended period. May be unconscious.",
                    severity="HIGH",
                )

        return None

    def _calculate_fall_confidence(
        self, is_horizontal: Optional[bool], hip_drop_vel: float, is_still: bool
    ) -> float:
        """Calculate confidence score for a fall event."""
        score = 0.0

        if is_horizontal is True:
            score += 0.4
        if hip_drop_vel > self._hip_drop_threshold:
            score += 0.3
        elif hip_drop_vel > self._hip_drop_threshold * 0.5:
            score += 0.15
        if is_still:
            score += 0.2

        score += 0.1

        return min(1.0, score)
