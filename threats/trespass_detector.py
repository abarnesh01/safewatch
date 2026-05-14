"""
SafeWatch Trespass Detector
Detects unauthorized entry into restricted zones.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from detection.person_detector import DetectedPerson
from detection.zone_manager import ZoneManager
from threats.fight_detector import ThreatEvent


class TrespassDetector:
    """Detects individuals entering restricted polygons or virtual fences."""

    def __init__(self, zone_manager: ZoneManager) -> None:
        self._zone_manager = zone_manager
        logger.info("TrespassDetector initialized")

    def detect(self, camera_id: str, 
               persons: List[DetectedPerson]) -> List[ThreatEvent]:
        """Check if any detected person is within a restricted zone."""
        events = []
        
        for p in persons:
            zone_name = self._zone_manager.is_in_zone(camera_id, p.center)
            
            if zone_name:
                # In this context, any zone presence is considered trespass if restricted
                # Usually managed by zone properties, here we simplify to zone existence
                events.append(ThreatEvent(
                    threat_type="trespass",
                    camera_id=camera_id,
                    severity="MEDIUM",
                    confidence=1.0,
                    description=f"Unauthorized entry detected in zone '{zone_name}' by person {p.person_id}",
                    person_ids=[p.person_id],
                    metadata={"zone_name": zone_name, "position": p.center}
                ))
        
        return events
