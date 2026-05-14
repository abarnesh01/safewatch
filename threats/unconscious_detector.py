"""
SafeWatch — UnconsciousDetector
Detects unconscious/unresponsive persons through extended horizontal stillness.
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


class UnconsciousDetector:
    """
    Detects unconscious or unresponsive persons.
    State machine: ACTIVE → FALLEN → POSSIBLY_UNCONSCIOUS → UNCONSCIOUS
    """

    def __init__(self, config: dict):
        self._config = config.get("threats", {}).get("unconscious", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.80)
        self._horizontal_threshold = self._config.get("horizontal_angle_threshold", 25)
        self._stillness_frames = self._config.get("stillness_frames", 90)

        passaway_config = config.get("threats", {}).get("pass_away", {})
        self._passaway_enabled = passaway_config.get("enabled", True)
        self._passaway_stillness = passaway_config.get("stillness_frames", 120)

        self._analyzer = SkeletonAnalyzer()
        self._person_states: dict[int, dict] = defaultdict(lambda: {
            "state": "ACTIVE",
            "horizontal_frames": 0,
            "stillness_frames": 0,
            "fall_detected": False,
            "alerted_possibly": False,
            "alerted_unconscious": False,
            "alerted_passaway": False,
            "last_update": time.time(),
        })
        logger.info(f"UnconsciousDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"UnconsciousDetector(enabled={self._enabled}, tracking={len(self._person_states)})"

    def detect(
        self,
        persons: list[Person],
        poses: list[PoseResult],
        velocity_tracker: VelocityTracker,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """Detect unconscious persons."""
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

        # Cleanup stale entries
        current_ids = {p.id for p in persons}
        stale = [pid for pid in self._person_states if pid not in current_ids]
        for pid in stale:
            s = self._person_states[pid]
            if time.time() - s["last_update"] > 30:
                del self._person_states[pid]

        return events

    def _evaluate_person(
        self,
        person: Person,
        pose: PoseResult,
        velocity_tracker: VelocityTracker,
    ) -> Optional[ThreatEvent]:
        """Evaluate unconscious state machine for a person."""
        state = self._person_states[person.id]
        state["last_update"] = time.time()

        is_horizontal = self._analyzer.is_person_horizontal(pose, self._horizontal_threshold)
        avg_vel = velocity_tracker.get_average_velocity(person.id)
        is_still = avg_vel < 3.0

        # Check head at ground level
        nose = pose.get_landmark("nose")
        head_low = False
        if nose is not None:
            head_low = nose["y"] > 0.7  # Nose near bottom of frame

        current = state["state"]

        if current == "ACTIVE":
            if is_horizontal is True:
                state["horizontal_frames"] += 1
                if state["horizontal_frames"] > 5:
                    state["state"] = "FALLEN"
                    state["stillness_frames"] = 0
                    state["fall_detected"] = True
                    logger.debug(f"Person {person.id}: ACTIVE → FALLEN")
            else:
                state["horizontal_frames"] = 0

        elif current == "FALLEN":
            if is_horizontal is True and is_still:
                state["stillness_frames"] += 1
            elif is_horizontal is True and not is_still:
                state["stillness_frames"] = max(0, state["stillness_frames"] - 1)
            else:
                # Person is no longer horizontal — recovering
                orientation = self._analyzer.get_body_orientation(pose)
                if orientation in ("standing", "sitting"):
                    state["state"] = "ACTIVE"
                    state["horizontal_frames"] = 0
                    state["stillness_frames"] = 0
                    return None

            if state["stillness_frames"] >= self._stillness_frames // 2:
                state["state"] = "POSSIBLY_UNCONSCIOUS"
                logger.debug(f"Person {person.id}: FALLEN → POSSIBLY_UNCONSCIOUS")

                if not state["alerted_possibly"]:
                    state["alerted_possibly"] = True
                    return ThreatEvent(
                        threat_type="UNCONSCIOUS",
                        confidence=round(min(0.85, self._confidence_threshold + 0.05), 3),
                        persons_involved=[person.id],
                        location_bbox=person.bbox,
                        description=(
                            f"Person may be unconscious. Horizontal and motionless for "
                            f"{state['stillness_frames']} frames."
                        ),
                        severity="HIGH",
                    )

        elif current == "POSSIBLY_UNCONSCIOUS":
            if is_horizontal is True and is_still:
                state["stillness_frames"] += 1
            elif not is_still or is_horizontal is False:
                orientation = self._analyzer.get_body_orientation(pose)
                if orientation in ("standing", "sitting"):
                    state["state"] = "ACTIVE"
                    state["horizontal_frames"] = 0
                    state["stillness_frames"] = 0
                    return None

            if state["stillness_frames"] >= self._stillness_frames:
                state["state"] = "UNCONSCIOUS"
                logger.debug(f"Person {person.id}: POSSIBLY_UNCONSCIOUS → UNCONSCIOUS")

                if not state["alerted_unconscious"]:
                    state["alerted_unconscious"] = True
                    return ThreatEvent(
                        threat_type="UNCONSCIOUS",
                        confidence=0.92,
                        persons_involved=[person.id],
                        location_bbox=person.bbox,
                        description=(
                            f"Person confirmed unconscious. Motionless on ground for "
                            f"{state['stillness_frames']} frames. Immediate attention required."
                        ),
                        severity="CRITICAL",
                    )

        elif current == "UNCONSCIOUS":
            if is_still:
                state["stillness_frames"] += 1
            else:
                orientation = self._analyzer.get_body_orientation(pose)
                if orientation in ("standing", "sitting"):
                    state["state"] = "ACTIVE"
                    state["horizontal_frames"] = 0
                    state["stillness_frames"] = 0
                    return None

            # Extended stillness — pass_away scenario
            if (self._passaway_enabled
                    and state["stillness_frames"] >= self._passaway_stillness
                    and not state["alerted_passaway"]):
                state["alerted_passaway"] = True
                return ThreatEvent(
                    threat_type="PASS_AWAY",
                    confidence=0.88,
                    persons_involved=[person.id],
                    location_bbox=person.bbox,
                    description=(
                        f"Extended unconsciousness detected. Person has been motionless for "
                        f"{state['stillness_frames']} frames. Emergency response needed."
                    ),
                    severity="CRITICAL",
                )

        return None

    def notify_fall_detected(self, person_id: int):
        """Notify that a fall was detected for this person (from FallDetector)."""
        state = self._person_states[person_id]
        state["fall_detected"] = True
