"""
SafeWatch Zone Manager
Handles spatial exclusion and inclusion zones for threat detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger


@dataclass
class Zone:
    """Geospatial or image-coordinate restricted area."""
    name: str
    polygon: List[Tuple[int, int]]
    camera_id: str
    is_exclusion: bool = False
    schedule: Dict[str, str] = field(default_factory=dict)  # {"start": "22:00", "end": "06:00"}


class ZoneManager:
    """Manages monitoring zones and point-in-polygon checks."""

    def __init__(self) -> None:
        self._zones: Dict[str, List[Zone]] = {}
        logger.info("ZoneManager initialized")

    def add_zone(self, zone: Zone) -> None:
        if zone.camera_id not in self._zones:
            self._zones[zone.camera_id] = []
        self._zones[zone.camera_id].append(zone)
        logger.info("Added zone '{}' for camera {}", zone.name, zone.camera_id)

    def is_in_zone(self, camera_id: str, point: Tuple[int, int]) -> Optional[str]:
        """Check if a point is within any active zone for a camera."""
        if camera_id not in self._zones:
            return None

        now = datetime.now().strftime("%H:%M")
        
        for zone in self._zones[camera_id]:
            # Schedule check
            if zone.schedule:
                start = zone.schedule.get("start", "00:00")
                end = zone.schedule.get("end", "23:59")
                if not (start <= now <= end):
                    continue

            # Polygon check
            poly = np.array(zone.polygon, dtype=np.int32)
            dist = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
            
            if dist >= 0:
                return zone.name
        
        return None

    def get_zones(self, camera_id: str) -> List[Zone]:
        return self._zones.get(camera_id, [])

    def draw_zones(self, frame: np.ndarray, camera_id: str) -> np.ndarray:
        """Render zone boundaries onto the frame."""
        annotated = frame.copy()
        zones = self.get_zones(camera_id)
        
        for zone in zones:
            color = (0, 0, 255) if zone.is_exclusion else (255, 255, 0)
            poly = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [poly], True, color, 2)
            
            # Label
            x, y = zone.polygon[0]
            cv2.putText(annotated, zone.name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return annotated
