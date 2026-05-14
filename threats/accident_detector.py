"""
SafeWatch — AccidentDetector
Detects accidents through multi-person falls, sudden motion spikes, and crowd dispersal.
"""

import time
from typing import Optional

from loguru import logger

from detection.person_detector import Person
from detection.pose_estimator import PoseResult
from detection.optical_flow import FlowResult
from classifier.velocity_tracker import VelocityTracker
from threats.fight_detector import ThreatEvent


class AccidentDetector:
    """Detects accident scenarios using fall clustering, motion spikes, and crowd patterns."""

    def __init__(self, config: dict):
        self._config = config.get("threats", {}).get("accident", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.78)
        self._recent_falls: list[dict] = []
        self._fall_window = 30
        self._last_alert_time = 0.0
        self._motion_spike_history: list[float] = []
        logger.info(f"AccidentDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"AccidentDetector(enabled={self._enabled}, recent_falls={len(self._recent_falls)})"

    def detect(
        self,
        persons: list[Person],
        poses: list[PoseResult],
        flow_result: Optional[FlowResult],
        velocity_tracker: VelocityTracker,
        fall_events: Optional[list] = None,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """Detect accident scenarios."""
        if not self._enabled:
            return []

        events = []
        now = time.time()

        # Track fall events for clustering
        if fall_events:
            for fe in fall_events:
                self._recent_falls.append({
                    "time": now,
                    "persons": fe.persons_involved,
                    "location": fe.location_bbox,
                })

        # Clean old falls outside window
        self._recent_falls = [
            f for f in self._recent_falls
            if now - f["time"] < self._fall_window
        ]

        score = 0.0

        # Signal 1: Multiple falls in short window
        if len(self._recent_falls) >= 2:
            score += 0.35
            if len(self._recent_falls) >= 3:
                score += 0.15

        # Signal 2: High optical flow spike followed by stillness
        if flow_result is not None:
            self._motion_spike_history.append(flow_result.max_magnitude)
            if len(self._motion_spike_history) > 30:
                self._motion_spike_history = self._motion_spike_history[-30:]

            if len(self._motion_spike_history) >= 10:
                earlier = self._motion_spike_history[-10:-5]
                recent = self._motion_spike_history[-5:]
                earlier_avg = sum(earlier) / len(earlier) if earlier else 0
                recent_avg = sum(recent) / len(recent) if recent else 0

                if earlier_avg > 30 and recent_avg < 5:
                    score += 0.25

        # Signal 3: Person count sudden change (people fleeing scene)
        active_count = len(persons)
        total_vel = sum(
            velocity_tracker.get_average_velocity(p.id) for p in persons
        )
        if active_count > 0 and total_vel / active_count > 20:
            score += 0.15

        # Signal 4: Falls clustered in same area
        if len(self._recent_falls) >= 2:
            locations = [f["location"] for f in self._recent_falls]
            centers = [
                ((l[0] + l[2]) / 2, (l[1] + l[3]) / 2)
                for l in locations
            ]
            max_dist = 0
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = (dx**2 + dy**2)**0.5
                    max_dist = max(max_dist, dist)

            if max_dist < 200:  # Falls in same area
                score += 0.15

        confidence = min(1.0, score)

        if (confidence >= self._confidence_threshold
                and now - self._last_alert_time > 30):
            self._last_alert_time = now

            all_person_ids = set()
            for f in self._recent_falls:
                all_person_ids.update(f["persons"])
            all_person_ids.update(p.id for p in persons)

            x1 = min(p.bbox[0] for p in persons) if persons else 0
            y1 = min(p.bbox[1] for p in persons) if persons else 0
            x2 = max(p.bbox[2] for p in persons) if persons else 640
            y2 = max(p.bbox[3] for p in persons) if persons else 480

            num_falls = len(self._recent_falls)
            severity = "CRITICAL" if num_falls >= 3 else "HIGH"

            events.append(ThreatEvent(
                threat_type="ACCIDENT",
                confidence=round(confidence, 3),
                persons_involved=list(all_person_ids),
                location_bbox=(x1, y1, x2, y2),
                description=(
                    f"Accident detected. {num_falls} falls in {self._fall_window}s window. "
                    f"{len(persons)} persons in area."
                ),
                severity=severity,
            ))

        return events
