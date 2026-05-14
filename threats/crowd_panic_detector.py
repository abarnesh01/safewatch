"""
SafeWatch Crowd Panic Detector
Detects sudden optical flow divergence and high-velocity crowd movement.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from detection.optical_flow import FlowStats
from threats.fight_detector import ThreatEvent


class CrowdPanicDetector:
    """Detects panic patterns in crowds using optical flow and velocity analysis."""

    def __init__(self, flow_divergence_thresh: float = 0.5,
                 acceleration_thresh: float = 10.0) -> None:
        self._divergence_thresh = flow_divergence_thresh
        self._accel_thresh = acceleration_thresh
        logger.info("CrowdPanicDetector initialized")

    def detect(self, camera_id: str, 
               flow_stats: Optional[FlowStats],
               person_count: int) -> List[ThreatEvent]:
        """Analyze crowd movement to identify panic or mass escape."""
        events = []
        
        if not flow_stats or person_count < 5:
            return events

        # High divergence + high magnitude = sudden directional movement
        if (flow_stats.divergence > self._divergence_thresh and 
            flow_stats.mean_magnitude > self._accel_thresh):
            
            events.append(ThreatEvent(
                threat_type="crowd_panic",
                camera_id=camera_id,
                severity="HIGH",
                confidence=float(min(flow_stats.divergence, 1.0)),
                description=f"Potential crowd panic detected: Rapid directional movement by {person_count} persons",
                metadata={"divergence": float(flow_stats.divergence), 
                          "magnitude": float(flow_stats.mean_magnitude),
                          "person_count": person_count}
            ))
        
        return events
