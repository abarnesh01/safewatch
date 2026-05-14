"""
SafeWatch Accident Detector
Detects simultaneous falls and sudden mass collapse patterns.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from threats.fight_detector import ThreatEvent


class AccidentDetector:
    """Detects multi-person accidents or simultaneous falls."""

    def __init__(self, simultaneous_fall_thresh: int = 2) -> None:
        self._simultaneous_thresh = simultaneous_fall_thresh
        logger.info("AccidentDetector initialized")

    def detect(self, camera_id: str, 
               active_threats: List[ThreatEvent]) -> List[ThreatEvent]:
        """Identify multi-person accident patterns from individual threat events."""
        events = []
        
        # Count simultaneous falls in the current frame's threat events
        fall_events = [e for e in active_threats if e.threat_type == "fall"]
        
        if len(fall_events) >= self._simultaneous_thresh:
            person_ids = []
            for e in fall_events:
                person_ids.extend(e.person_ids)
            
            events.append(ThreatEvent(
                threat_type="accident",
                camera_id=camera_id,
                severity="HIGH",
                confidence=0.9,
                description=f"Mass accident detected: {len(fall_events)} simultaneous falls",
                person_ids=list(set(person_ids)),
                metadata={"fall_count": len(fall_events)}
            ))
        
        return events
