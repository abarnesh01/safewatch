"""
SafeWatch Fight Detector
Detects aggressive physical altercations between multiple persons.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from detection.person_detector import DetectedPerson
from detection.pose_estimator import PersonPose


@dataclass
class ThreatEvent:
    """Represents a detected threat event."""
    threat_type: str
    camera_id: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    person_ids: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class FightDetector:
    """Detects fighting behavior based on proximity and kinetic energy."""

    def __init__(self, proximity_threshold: float = 150.0,
                 aggression_score_threshold: float = 0.6) -> None:
        self._proximity_thresh = proximity_threshold
        self._aggression_thresh = aggression_score_threshold
        logger.info("FightDetector initialized (proximity_thresh={})", proximity_threshold)

    def detect(self, camera_id: str, 
               persons: List[DetectedPerson], 
               poses: Dict[int, PersonPose],
               velocities: Dict[int, Dict[str, float]]) -> List[ThreatEvent]:
        """Analyze person interactions to identify fighting."""
        events = []
        
        if len(persons) < 2:
            return events

        # Check pairwise interactions
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1, p2 = persons[i], persons[j]
                
                # Proximity check
                dist = np.sqrt((p1.center[0] - p2.center[0])**2 + 
                              (p1.center[1] - p2.center[1])**2)
                
                if dist < self._proximity_thresh:
                    # Potential interaction, analyze kinetics
                    aggression_score = self._compute_aggression(p1, p2, poses, velocities)
                    
                    if aggression_score > self._aggression_thresh:
                        severity = "HIGH" if aggression_score > 0.8 else "MEDIUM"
                        events.append(ThreatEvent(
                            threat_type="fight",
                            camera_id=camera_id,
                            severity=severity,
                            confidence=float(aggression_score),
                            description=f"Physical altercation detected between person {p1.person_id} and {p2.person_id}",
                            person_ids=[p1.person_id, p2.person_id],
                            metadata={"distance": float(dist), "aggression_score": float(aggression_score)}
                        ))
        
        return events

    def _compute_aggression(self, p1: DetectedPerson, p2: DetectedPerson,
                            poses: Dict[int, PersonPose],
                            velocities: Dict[int, Dict[str, float]]) -> float:
        """Compute an aggression score based on movement and pose."""
        score = 0.0
        
        v1 = velocities.get(p1.person_id, {}).get("person_velocity", 0.0)
        v2 = velocities.get(p2.person_id, {}).get("person_velocity", 0.0)
        
        vh1 = max(velocities.get(p1.person_id, {}).get("left_wrist_velocity", 0.0),
                  velocities.get(p1.person_id, {}).get("right_wrist_velocity", 0.0))
        vh2 = max(velocities.get(p2.person_id, {}).get("left_wrist_velocity", 0.0),
                  velocities.get(p2.person_id, {}).get("right_wrist_velocity", 0.0))

        # High hand velocity near someone else is suspicious
        if vh1 > 25.0: score += 0.4
        if vh2 > 25.0: score += 0.4
        
        # Rapid movement towards each other
        if v1 > 10.0 and v2 > 10.0: score += 0.2
        
        return min(score, 1.0)
