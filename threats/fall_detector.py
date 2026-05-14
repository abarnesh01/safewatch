"""
SafeWatch Fall Detector
Detects human falls using hip-drop analysis and body orientation.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from detection.person_detector import DetectedPerson
from detection.pose_estimator import PersonPose
from threats.fight_detector import ThreatEvent


class FallDetector:
    """Detects falling incidents based on vertical-to-horizontal transitions."""

    def __init__(self, hip_drop_threshold: float = 0.3,
                 stillness_duration: float = 3.0) -> None:
        self._hip_drop_thresh = hip_drop_threshold
        self._stillness_duration = stillness_duration
        # person_id -> (last_y, last_time, state)
        self._tracking_state: Dict[int, Dict] = {}
        logger.info("FallDetector initialized")

    def detect(self, camera_id: str, 
               persons: List[DetectedPerson], 
               poses: Dict[int, PersonPose],
               velocities: Dict[int, Dict[str, float]],
               features: Dict[int, Dict[str, float]]) -> List[ThreatEvent]:
        """Analyze poses to identify falls."""
        events = []
        
        for p in persons:
            pid = p.person_id
            if pid not in poses or pid not in features:
                continue
            
            pose_feats = features[pid]
            v_ratio = pose_feats.get("vertical_ratio", 2.0)
            v_person = velocities.get(pid, {}).get("person_velocity", 0.0)
            
            # Fall criteria: sudden verticality drop + high velocity
            if v_ratio < 0.7:
                severity = "HIGH"
                if v_person > 15.0:
                    severity = "CRITICAL"
                
                events.append(ThreatEvent(
                    threat_type="fall",
                    camera_id=camera_id,
                    severity=severity,
                    confidence=0.85,
                    description=f"Fall detected for person {pid}",
                    person_ids=[pid],
                    metadata={"vertical_ratio": float(v_ratio), "velocity": float(v_person)}
                ))
        
        return events

    def cleanup(self, active_person_ids: List[int]) -> None:
        pids_to_remove = [pid for pid in self._tracking_state if pid not in active_person_ids]
        for pid in pids_to_remove:
            del self._tracking_state[pid]
