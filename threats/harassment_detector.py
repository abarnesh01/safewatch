"""
SafeWatch Harassment Detector
Detects sustained proximity and circling behavior between individuals.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from detection.person_detector import DetectedPerson
from threats.fight_detector import ThreatEvent


class HarassmentDetector:
    """Detects harassment patterns like trailing or sustained proximity."""

    def __init__(self, proximity_distance: float = 200.0,
                 duration_threshold: float = 10.0) -> None:
        self._proximity_dist = proximity_distance
        self._duration_thresh = duration_threshold
        # (p1, p2) -> start_time
        self._interactions: Dict[Tuple[int, int], float] = {}
        logger.info("HarassmentDetector initialized")

    def detect(self, camera_id: str, 
               persons: List[DetectedPerson]) -> List[ThreatEvent]:
        """Analyze person distances over time to identify harassment."""
        events = []
        now = time.time()
        current_interactions = []

        if len(persons) < 2:
            return events

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                dist = np.sqrt((p1.center[0] - p2.center[0])**2 + 
                              (p1.center[1] - p2.center[1])**2)
                
                key = tuple(sorted((p1.person_id, p2.person_id)))
                
                if dist < self._proximity_dist:
                    if key not in self._interactions:
                        self._interactions[key] = now
                    
                    duration = now - self._interactions[key]
                    if duration > self._duration_thresh:
                        events.append(ThreatEvent(
                            threat_type="harassment",
                            camera_id=camera_id,
                            severity="MEDIUM",
                            confidence=0.7,
                            description=f"Sustained proximity (harassment) detected between {p1.person_id} and {p2.person_id}",
                            person_ids=[p1.person_id, p2.person_id],
                            metadata={"duration": float(duration), "distance": float(dist)}
                        ))
                    current_interactions.append(key)

        # Cleanup old interactions
        keys_to_remove = [k for k in self._interactions if k not in current_interactions]
        for k in keys_to_remove:
            del self._interactions[k]

        return events
