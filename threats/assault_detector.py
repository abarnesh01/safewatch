"""
SafeWatch Assault Detector
Detects physical assault using strike trajectory and evasion analysis.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from detection.person_detector import DetectedPerson
from detection.pose_estimator import PersonPose
from threats.fight_detector import ThreatEvent


class AssaultDetector:
    """Detects asymmetric physical assault based on rapid kinetic strikes."""

    def __init__(self, strike_velocity_thresh: float = 20.0) -> None:
        self._strike_thresh = strike_velocity_thresh
        logger.info("AssaultDetector initialized")

    def detect(self, camera_id: str, 
               persons: List[DetectedPerson],
               velocities: Dict[int, Dict[str, float]]) -> List[ThreatEvent]:
        """Identify potential assault based on sudden high-velocity movements."""
        events = []
        
        for p in persons:
            pid = p.person_id
            if pid not in velocities:
                continue
            
            # Check for high-velocity limb movement
            v_l_hand = velocities[pid].get("left_wrist_velocity", 0.0)
            v_r_hand = velocities[pid].get("right_wrist_velocity", 0.0)
            v_max_hand = max(v_l_hand, v_r_hand)

            if v_max_hand > self._strike_thresh:
                # Potential strike, check for nearby victim
                for other in persons:
                    if other.person_id == pid:
                        continue
                    
                    dist = np.sqrt((p.center[0] - other.center[0])**2 + 
                                  (p.center[1] - other.center[1])**2)
                    
                    if dist < 150.0:
                        events.append(ThreatEvent(
                            threat_type="assault",
                            camera_id=camera_id,
                            severity="CRITICAL",
                            confidence=0.8,
                            description=f"Potential assault detected: Person {pid} striking towards Person {other.person_id}",
                            person_ids=[pid, other.person_id],
                            metadata={"strike_velocity": float(v_max_hand), "distance": float(dist)}
                        ))
        
        return events
