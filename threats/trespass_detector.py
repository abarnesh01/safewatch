"""
SafeWatch — TrespassDetector
Detects unauthorized entry into restricted polygon zones.
"""

import time
from collections import defaultdict
from typing import Optional

from loguru import logger

from detection.person_detector import Person
from detection.zone_manager import ZoneManager
from threats.fight_detector import ThreatEvent


class TrespassDetector:
    """Detects persons entering restricted zones with dwell-time tracking."""

    def __init__(self, config: dict, zone_manager: ZoneManager):
        self._config = config.get("threats", {}).get("trespass", {})
        self._enabled = self._config.get("enabled", True)
        self._confidence_threshold = self._config.get("confidence_threshold", 0.95)
        self._zone_manager = zone_manager
        self._dwell_tracking: dict[tuple, dict] = defaultdict(lambda: {
            "entry_time": 0.0,
            "dwell_frames": 0,
            "alerted_entry": False,
            "alerted_dwell": False,
        })
        self._critical_dwell_seconds = 5.0
        logger.info(f"TrespassDetector initialized (enabled={self._enabled})")

    def __repr__(self) -> str:
        return f"TrespassDetector(enabled={self._enabled}, zones={len(self._zone_manager.get_all_zones())})"

    def detect(
        self,
        persons: list,
        poses: list = None,
        velocity_tracker=None,
        config: Optional[dict] = None,
    ) -> list[ThreatEvent]:
        """Detect trespass violations."""
        if not self._enabled:
            return []

        events = []
        violations = self._zone_manager.get_violations(persons)

        active_keys = set()

        for violation in violations:
            person = violation["person"]
            zone_name = violation["zone_name"]
            zone_type = violation["zone_type"]
            key = (person.id, zone_name)
            active_keys.add(key)

            state = self._dwell_tracking[key]

            if state["entry_time"] == 0:
                state["entry_time"] = time.time()

            state["dwell_frames"] += 1
            dwell_seconds = time.time() - state["entry_time"]

            # Immediate alert for critical zones
            if zone_type == "high_security" and not state["alerted_entry"]:
                state["alerted_entry"] = True
                events.append(ThreatEvent(
                    threat_type="TRESPASS",
                    confidence=self._confidence_threshold,
                    persons_involved=[person.id],
                    location_bbox=person.bbox,
                    description=(
                        f"CRITICAL: Person {person.id} entered high-security zone '{zone_name}'. "
                        f"Immediate attention required."
                    ),
                    severity="CRITICAL",
                ))

            # First entry alert for restricted zones
            elif not state["alerted_entry"] and state["dwell_frames"] > 3:
                state["alerted_entry"] = True
                events.append(ThreatEvent(
                    threat_type="TRESPASS",
                    confidence=self._confidence_threshold,
                    persons_involved=[person.id],
                    location_bbox=person.bbox,
                    description=(
                        f"Person {person.id} entered restricted zone '{zone_name}' "
                        f"(type: {zone_type})."
                    ),
                    severity="MEDIUM",
                ))

            # Extended dwell alert
            if (dwell_seconds > self._critical_dwell_seconds
                    and not state["alerted_dwell"]):
                state["alerted_dwell"] = True
                events.append(ThreatEvent(
                    threat_type="TRESPASS",
                    confidence=self._confidence_threshold,
                    persons_involved=[person.id],
                    location_bbox=person.bbox,
                    description=(
                        f"Person {person.id} has been in restricted zone '{zone_name}' "
                        f"for {dwell_seconds:.1f}s. Extended presence detected."
                    ),
                    severity="HIGH",
                ))

        # Reset tracking for persons who left zones
        stale = [k for k in self._dwell_tracking if k not in active_keys]
        for k in stale:
            if time.time() - self._dwell_tracking[k]["entry_time"] > 30:
                del self._dwell_tracking[k]

        return events
