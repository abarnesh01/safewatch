"""
SafeWatch — FightDetector
Detects physical fights between multiple people using velocity, proximity, and pose analysis.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from detection.person_detector import Person
from detection.pose_estimator import PoseResult
from classifier.skeleton_analyzer import SkeletonAnalyzer
from classifier.velocity_tracker import VelocityTracker


@dataclass
class ThreatEvent:
    """Represents a detected threat event."""
    threat_type: str
    confidence: float
    persons_involved: list[int]
    location_bbox: tuple[int, int, int, int]
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: str = ""

    def __repr__(self) -> str:
        return (
            f"ThreatEvent(type='{self.threat_type}', confidence={self.confidence:.2f}, "
            f"severity='{self.severity}', persons={self.persons_involved})"
        )


class FightDetector:
    """Detects physical fights between two or more people."""

    def __init__(self, config: dict):
        self._config = config.get("threats", {}).get("fight", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.82)
        self._min_persons = self._config.get("min_persons", 2)
        self._velocity_threshold = self._config.get("velocity_threshold", 45.0)
        self._overlap_threshold = self._config.get("overlap_threshold", 0.3)
        self._analyzer = SkeletonAnalyzer()
        logger.info(f"FightDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"FightDetector(enabled={self._enabled}, threshold={self._confidence_threshold})"

    def detect(
        self,
        persons: list[Person],
        poses: list[PoseResult],
        velocity_tracker: VelocityTracker,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """
        Detect fight scenarios.

        Returns:
            List of ThreatEvent objects for detected fights.
        """
        if not self._enabled or len(persons) < self._min_persons:
            return []

        events = []
        pose_map = {p.person_id: p for p in poses}

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                confidence = self._evaluate_fight_pair(
                    p1, p2, pose_map, velocity_tracker
                )

                if confidence >= self._confidence_threshold:
                    x1 = min(p1.bbox[0], p2.bbox[0])
                    y1 = min(p1.bbox[1], p2.bbox[1])
                    x2 = max(p1.bbox[2], p2.bbox[2])
                    y2 = max(p1.bbox[3], p2.bbox[3])

                    severity = "HIGH" if confidence > 0.9 else "MEDIUM"
                    if len(persons) > 3:
                        severity = "CRITICAL"

                    events.append(ThreatEvent(
                        threat_type="FIGHT",
                        confidence=round(confidence, 3),
                        persons_involved=[p1.id, p2.id],
                        location_bbox=(x1, y1, x2, y2),
                        description=(
                            f"Physical fight detected between {len(set([p1.id, p2.id]))} persons. "
                            f"High velocity aggressive movements observed."
                        ),
                        severity=severity,
                    ))

        return events

    def _evaluate_fight_pair(
        self,
        p1: Person,
        p2: Person,
        pose_map: dict[int, PoseResult],
        velocity_tracker: VelocityTracker,
    ) -> float:
        """Score the likelihood of a fight between two persons."""
        score = 0.0
        max_signals = 5

        # Signal 1: Proximity
        dist = np.sqrt((p1.center[0] - p2.center[0])**2 + (p1.center[1] - p2.center[1])**2)
        avg_width = (p1.width + p2.width) / 2
        if avg_width > 0 and dist < avg_width * 1.5:
            score += 1.0

        # Signal 2: Relative velocity (approaching each other fast)
        rel_vel = velocity_tracker.get_relative_velocity(p1.id, p2.id)
        if rel_vel > self._velocity_threshold:
            score += 1.0
        elif rel_vel > self._velocity_threshold * 0.5:
            score += 0.5

        # Signal 3: Arm raise level (aggressive posture)
        pose1 = pose_map.get(p1.id)
        pose2 = pose_map.get(p2.id)

        if pose1 is not None:
            arm1 = self._analyzer.get_arm_raise_level(pose1)
            if arm1 is not None and arm1 > 0.5:
                score += 0.5

        if pose2 is not None:
            arm2 = self._analyzer.get_arm_raise_level(pose2)
            if arm2 is not None and arm2 > 0.5:
                score += 0.5

        # Signal 4: Wrist velocity (striking motions)
        wrist_vel1 = velocity_tracker.get_velocity(p1.id, "left_wrist")
        wrist_vel2 = velocity_tracker.get_velocity(p1.id, "right_wrist")
        wrist_vel3 = velocity_tracker.get_velocity(p2.id, "left_wrist")
        wrist_vel4 = velocity_tracker.get_velocity(p2.id, "right_wrist")

        max_wrist = max(wrist_vel1, wrist_vel2, wrist_vel3, wrist_vel4)
        if max_wrist > self._velocity_threshold * 1.2:
            score += 1.0
        elif max_wrist > self._velocity_threshold * 0.6:
            score += 0.5

        # Signal 5: Body lean toward opponent
        if pose1 is not None:
            lean1 = self._analyzer.get_body_lean_angle(pose1)
            if lean1 is not None and lean1 > 15:
                score += 0.5

        if pose2 is not None:
            lean2 = self._analyzer.get_body_lean_angle(pose2)
            if lean2 is not None and lean2 > 15:
                score += 0.5

        confidence = min(1.0, score / max_signals)
        return confidence
