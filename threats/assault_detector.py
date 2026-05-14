"""
SafeWatch — AssaultDetector
Detects one-sided physical assault with victim/assailant role assignment.
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


class AssaultDetector:
    """Detects physical assault — more aggressive/one-sided than fight detection."""

    def __init__(self, config: dict):
        self._config = config.get("threats", {}).get("assault", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.85)
        self._strike_velocity_threshold = self._config.get("strike_velocity_threshold", 60.0)
        self._analyzer = SkeletonAnalyzer()
        self._pair_state: dict[tuple, dict] = defaultdict(lambda: {
            "strike_count": 0,
            "assailant_id": None,
            "victim_id": None,
            "last_strike_time": 0.0,
            "alerted": False,
        })
        logger.info(f"AssaultDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"AssaultDetector(enabled={self._enabled}, threshold={self._confidence_threshold})"

    def detect(
        self,
        persons: list[Person],
        poses: list[PoseResult],
        velocity_tracker: VelocityTracker,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """Detect assault scenarios."""
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

        return events

    def _evaluate_pair(
        self,
        p1: Person,
        p2: Person,
        pose_map: dict[int, PoseResult],
        velocity_tracker: VelocityTracker,
    ) -> Optional[ThreatEvent]:
        """Evaluate assault between a pair of persons."""
        pair_key = (min(p1.id, p2.id), max(p1.id, p2.id))
        state = self._pair_state[pair_key]

        dist = np.sqrt(
            (p1.center[0] - p2.center[0])**2 +
            (p1.center[1] - p2.center[1])**2
        )

        if dist > max(p1.width, p2.width) * 2.5:
            return None

        pose1 = pose_map.get(p1.id)
        pose2 = pose_map.get(p2.id)

        # Determine who is the assailant (higher wrist velocity directed at other)
        wrist_vel_p1 = max(
            velocity_tracker.get_velocity(p1.id, "left_wrist"),
            velocity_tracker.get_velocity(p1.id, "right_wrist"),
        )
        wrist_vel_p2 = max(
            velocity_tracker.get_velocity(p2.id, "left_wrist"),
            velocity_tracker.get_velocity(p2.id, "right_wrist"),
        )

        body_vel_p1 = velocity_tracker.get_average_velocity(p1.id)
        body_vel_p2 = velocity_tracker.get_average_velocity(p2.id)

        # Assign roles
        if wrist_vel_p1 > wrist_vel_p2 * 1.5:
            assailant, victim = p1, p2
            assailant_pose = pose1
            victim_pose = pose2
            strike_vel = wrist_vel_p1
        elif wrist_vel_p2 > wrist_vel_p1 * 1.5:
            assailant, victim = p2, p1
            assailant_pose = pose2
            victim_pose = pose1
            strike_vel = wrist_vel_p2
        else:
            return None  # No clear asymmetry — might be mutual fight, not assault

        score = 0.0

        # Signal 1: Assailant has high wrist velocity (striking)
        if strike_vel > self._strike_velocity_threshold:
            score += 0.3
        elif strike_vel > self._strike_velocity_threshold * 0.6:
            score += 0.15

        # Signal 2: Victim is retreating or stationary
        victim_vel = velocity_tracker.get_average_velocity(victim.id)
        rel_vel = velocity_tracker.get_relative_velocity(assailant.id, victim.id)

        if victim_vel < 5 and rel_vel > 10:
            score += 0.2  # Victim stationary, assailant approaching

        # Signal 3: Victim is cowering (low body angle)
        if victim_pose is not None:
            lean = self._analyzer.get_body_lean_angle(victim_pose)
            if lean is not None and lean > 25:
                score += 0.15  # Victim ducking/cowering

            orientation = self._analyzer.get_body_orientation(victim_pose)
            if orientation == "crouching":
                score += 0.1

        # Signal 4: Assailant has aggressive posture
        if assailant_pose is not None:
            arm_level = self._analyzer.get_arm_raise_level(assailant_pose)
            if arm_level is not None and arm_level > 0.6:
                score += 0.15

        # Signal 5: Repeated strike motions
        if strike_vel > self._strike_velocity_threshold * 0.5:
            now = time.time()
            if now - state["last_strike_time"] < 2.0:
                state["strike_count"] += 1
            else:
                state["strike_count"] = 1
            state["last_strike_time"] = now

        if state["strike_count"] >= 3:
            score += 0.2

        confidence = min(1.0, score)

        if confidence >= self._confidence_threshold and not state["alerted"]:
            state["alerted"] = True
            state["assailant_id"] = assailant.id
            state["victim_id"] = victim.id

            x1 = min(assailant.bbox[0], victim.bbox[0])
            y1 = min(assailant.bbox[1], victim.bbox[1])
            x2 = max(assailant.bbox[2], victim.bbox[2])
            y2 = max(assailant.bbox[3], victim.bbox[3])

            return ThreatEvent(
                threat_type="ASSAULT",
                confidence=round(confidence, 3),
                persons_involved=[assailant.id, victim.id],
                location_bbox=(x1, y1, x2, y2),
                description=(
                    f"Physical assault detected. Person {assailant.id} assaulting Person {victim.id}. "
                    f"Strike velocity: {strike_vel:.1f} px/s. Strike count: {state['strike_count']}."
                ),
                severity="CRITICAL",
            )

        return None
