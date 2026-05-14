"""
SafeWatch — CrowdPanicDetector
Detects crowd panic patterns using optical flow divergence and multi-person analysis.
"""

import time
from typing import Optional

from loguru import logger

from detection.person_detector import Person
from detection.optical_flow import FlowResult
from classifier.velocity_tracker import VelocityTracker
from threats.fight_detector import ThreatEvent


class CrowdPanicDetector:
    """Detects crowd panic through optical flow divergence and mass movement patterns."""

    def __init__(self, config: dict):
        self._config = config.get("threats", {}).get("crowd_panic", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.72)
        self._flow_divergence_threshold = self._config.get("flow_divergence_threshold", 8.0)
        self._min_persons = self._config.get("min_persons", 5)
        self._last_alert_time = 0.0
        self._alert_cooldown = 30.0
        self._prev_person_count = 0
        self._person_count_history: list[int] = []
        logger.info(f"CrowdPanicDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"CrowdPanicDetector(enabled={self._enabled}, min_persons={self._min_persons})"

    def detect(
        self,
        persons: list[Person],
        flow_result: Optional[FlowResult],
        velocity_tracker: VelocityTracker,
        fall_events: Optional[list] = None,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """Detect crowd panic scenarios."""
        if not self._enabled:
            return []

        if len(persons) < self._min_persons:
            self._person_count_history.append(len(persons))
            if len(self._person_count_history) > 60:
                self._person_count_history = self._person_count_history[-60:]
            return []

        events = []
        score = 0.0

        # Signal 1: Optical flow divergence
        if flow_result is not None:
            if flow_result.divergence_score > self._flow_divergence_threshold:
                score += 0.35
            elif flow_result.divergence_score > self._flow_divergence_threshold * 0.6:
                score += 0.15

            if flow_result.mean_magnitude > 20:
                score += 0.15

        # Signal 2: Sudden person count change (crowd rushing in/out)
        self._person_count_history.append(len(persons))
        if len(self._person_count_history) > 60:
            self._person_count_history = self._person_count_history[-60:]

        if len(self._person_count_history) >= 10:
            recent = self._person_count_history[-5:]
            earlier = self._person_count_history[-10:-5]
            avg_recent = sum(recent) / len(recent)
            avg_earlier = sum(earlier) / len(earlier)

            if avg_earlier > 0:
                change_ratio = abs(avg_recent - avg_earlier) / avg_earlier
                if change_ratio > 0.5:
                    score += 0.2

        # Signal 3: High average velocity of all persons (everyone running)
        velocities = []
        for person in persons:
            vel = velocity_tracker.get_average_velocity(person.id)
            velocities.append(vel)

        if velocities:
            avg_vel = sum(velocities) / len(velocities)
            if avg_vel > 30:
                score += 0.2
            elif avg_vel > 15:
                score += 0.1

        # Signal 4: Multiple simultaneous falls
        if fall_events is not None and len(fall_events) >= 2:
            score += 0.2

        # Signal 5: People moving away from common center
        if len(persons) >= self._min_persons:
            centers = [(p.center[0], p.center[1]) for p in persons]
            cx = sum(c[0] for c in centers) / len(centers)
            cy = sum(c[1] for c in centers) / len(centers)

            moving_away = 0
            for person in persons:
                trajectory = velocity_tracker.get_trajectory(person.id, 5)
                if len(trajectory) >= 2:
                    start = trajectory[0]
                    end = trajectory[-1]
                    d_start = ((start[0] - cx)**2 + (start[1] - cy)**2)**0.5
                    d_end = ((end[0] - cx)**2 + (end[1] - cy)**2)**0.5
                    if d_end > d_start * 1.1:
                        moving_away += 1

            if len(persons) > 0:
                away_ratio = moving_away / len(persons)
                if away_ratio > 0.6:
                    score += 0.15

        confidence = min(1.0, score)
        now = time.time()

        if (confidence >= self._confidence_threshold
                and now - self._last_alert_time > self._alert_cooldown):
            self._last_alert_time = now

            all_ids = [p.id for p in persons]
            all_bboxes = [p.bbox for p in persons]
            x1 = min(b[0] for b in all_bboxes)
            y1 = min(b[1] for b in all_bboxes)
            x2 = max(b[2] for b in all_bboxes)
            y2 = max(b[3] for b in all_bboxes)

            severity = "CRITICAL" if confidence > 0.85 else "HIGH"

            events.append(ThreatEvent(
                threat_type="CROWD_PANIC",
                confidence=round(confidence, 3),
                persons_involved=all_ids,
                location_bbox=(x1, y1, x2, y2),
                description=(
                    f"Crowd panic detected with {len(persons)} persons. "
                    f"Flow divergence: {flow_result.divergence_score:.1f}. "
                    f"People moving in scattered directions."
                ),
                severity=severity,
            ))

        return events
