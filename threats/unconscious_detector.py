"""
SafeWatch Unconscious Detector
Detects prolonged horizontal posture with near-zero motion.
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


class UnconsciousDetector:
    """Detects individuals who may be unconscious or incapacitated."""

    def __init__(self, horizontal_duration_thresh: float = 15.0,
                 motion_threshold: float = 1.0) -> None:
        self._duration_thresh = horizontal_duration_thresh
        self._motion_thresh = motion_threshold
        # person_id -> horizontal_start_time
        self._horizontal_timers: Dict[int, float] = {}
        logger.info("UnconsciousDetector initialized")

    def detect(self, camera_id: str, 
               persons: List[DetectedPerson],
               features: Dict[int, Dict[str, float]],
               velocities: Dict[int, Dict[str, float]]) -> List[ThreatEvent]:
        """Identify potential unconsciousness based on posture and stillness."""
        events = []
        now = time.time()
        active_ids = []

        for p in persons:
            pid = p.person_id
            active_ids.append(pid)
            
            if pid not in features:
                continue
            
            v_ratio = features[pid].get("vertical_ratio", 2.0)
            v_person = velocities.get(pid, {}).get("person_velocity", 0.0)

            # Criteria: horizontal posture + very low movement
            if v_ratio < 0.6 and v_person < self._motion_thresh:
                if pid not in self._horizontal_timers:
                    self._horizontal_timers[pid] = now
                
                duration = now - self._horizontal_timers[pid]
                if duration > self._duration_thresh:
                    events.append(ThreatEvent(
                        threat_type="unconscious",
                        camera_id=camera_id,
                        severity="CRITICAL",
                        confidence=0.9,
                        description=f"Person {pid} detected horizontal and still for {int(duration)}s (potential unconsciousness)",
                        person_ids=[pid],
                        metadata={"duration": float(duration), "vertical_ratio": float(v_ratio), "velocity": float(v_person)}
                    ))
            else:
                # Person is upright or moving, reset timer
                if pid in self._horizontal_timers:
                    del self._horizontal_timers[pid]

        # Cleanup timers for persons no longer in view
        keys_to_remove = [pid for pid in self._horizontal_timers if pid not in active_ids]
        for pid in keys_to_remove:
            del self._horizontal_timers[pid]

        return events
