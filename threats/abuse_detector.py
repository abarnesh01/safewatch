"""
SafeWatch Abuse Detector
Detects repeated aggression and dominant/submissive posture relationships.
"""

import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from threats.fight_detector import ThreatEvent


class AbuseDetector:
    """Detects sustained abuse patterns using historical threat aggregation."""

    def __init__(self, strike_window: float = 60.0,
                 strike_count_thresh: int = 5) -> None:
        self._window = strike_window
        self._thresh = strike_count_thresh
        # (attacker_id, victim_id) -> [timestamps]
        self._history: Dict[Tuple[int, int], List[float]] = {}
        logger.info("AbuseDetector initialized")

    def detect(self, camera_id: str, 
               active_threats: List[ThreatEvent]) -> List[ThreatEvent]:
        """Analyze historical aggression to identify abuse patterns."""
        events = []
        now = time.time()
        
        # Track assaults/fights to build history
        for e in active_threats:
            if e.threat_type in ["assault", "fight"] and len(e.person_ids) >= 2:
                # Assume first is attacker, second is victim for history tracking
                # In a real system, this would be more sophisticated
                key = (e.person_ids[0], e.person_ids[1])
                if key not in self._history:
                    self._history[key] = []
                self._history[key].append(now)

        # Check for abuse patterns
        for key, timestamps in list(self._history.items()):
            # Filter timestamps within window
            valid_ts = [ts for ts in timestamps if now - ts < self._window]
            self._history[key] = valid_ts
            
            if len(valid_ts) >= self._thresh:
                events.append(ThreatEvent(
                    threat_type="abuse",
                    camera_id=camera_id,
                    severity="CRITICAL",
                    confidence=0.85,
                    description=f"Sustained abuse pattern detected between person {key[0]} and {key[1]}",
                    person_ids=list(key),
                    metadata={"incident_count": len(valid_ts), "window_seconds": self._window}
                ))
                # Clear history for this pair after detection to prevent alert flooding
                del self._history[key]
        
        return events

    def cleanup(self) -> None:
        """Clear old history."""
        now = time.time()
        for key in list(self._history.keys()):
            self._history[key] = [ts for ts in self._history[key] if now - ts < self._window]
            if not self._history[key]:
                del self._history[key]
