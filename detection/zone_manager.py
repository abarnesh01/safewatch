"""
SafeWatch — ZoneManager
Manages polygon-based restricted zones for trespass detection.
"""

import json
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from detection.person_detector import Person


class ZoneManager:
    """
    Manages named polygon zones for restricted area monitoring.
    Supports point-in-polygon testing, zone drawing, and runtime zone updates.
    """

    def __init__(self, config: dict):
        self._config = config
        self._zones: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._zones_file = Path("config_zones.json")
        self._load_zones_from_config()
        logger.info(f"ZoneManager initialized with {len(self._zones)} zones")

    def __repr__(self) -> str:
        return f"ZoneManager(zones={list(self._zones.keys())})"

    def _load_zones_from_config(self):
        """Load zones from config and persisted zone file."""
        threat_config = self._config.get("threats", {})
        trespass_config = threat_config.get("trespass", {})
        config_zones = trespass_config.get("zones", [])

        for zone_data in config_zones:
            if isinstance(zone_data, dict):
                name = zone_data.get("name", "unnamed")
                points = zone_data.get("points", [])
                zone_type = zone_data.get("type", "restricted")
                if points:
                    self._zones[name] = {
                        "name": name,
                        "points": points,
                        "type": zone_type,
                        "polygon": np.array(points, dtype=np.int32),
                    }

        if self._zones_file.exists():
            try:
                with open(self._zones_file, "r") as f:
                    saved_zones = json.load(f)
                for name, zone_data in saved_zones.items():
                    points = zone_data.get("points", [])
                    zone_type = zone_data.get("type", "restricted")
                    if points:
                        self._zones[name] = {
                            "name": name,
                            "points": points,
                            "type": zone_type,
                            "polygon": np.array(points, dtype=np.int32),
                        }
                logger.info(f"Loaded {len(saved_zones)} zones from {self._zones_file}")
            except Exception as e:
                logger.warning(f"Failed to load saved zones: {e}")

    def add_zone(self, name: str, polygon_points: list[list[int]], zone_type: str = "restricted"):
        """
        Add a new zone.

        Args:
            name: Unique zone name
            polygon_points: List of [x, y] coordinate pairs forming the polygon
            zone_type: Type of zone ("restricted", "high_security", "entrance", etc.)
        """
        with self._lock:
            self._zones[name] = {
                "name": name,
                "points": polygon_points,
                "type": zone_type,
                "polygon": np.array(polygon_points, dtype=np.int32),
            }
            self._save_zones()
            logger.info(f"Zone added: {name} ({zone_type}) with {len(polygon_points)} points")

    def remove_zone(self, name: str) -> bool:
        """Remove a zone by name."""
        with self._lock:
            if name in self._zones:
                del self._zones[name]
                self._save_zones()
                logger.info(f"Zone removed: {name}")
                return True
            return False

    def is_in_zone(self, point: tuple[int, int], zone_name: str) -> bool:
        """
        Check if a point is inside a named zone.

        Args:
            point: (x, y) pixel coordinates
            zone_name: Name of the zone to check

        Returns:
            True if the point is inside the zone polygon
        """
        with self._lock:
            zone = self._zones.get(zone_name)
        if zone is None:
            return False

        result = cv2.pointPolygonTest(zone["polygon"], point, False)
        return result >= 0

    def get_violations(self, persons: list[Person], zone_name: Optional[str] = None) -> list[dict]:
        """
        Find all persons violating restricted zones.

        Args:
            persons: List of detected Person objects
            zone_name: Optional specific zone to check. If None, checks all zones.

        Returns:
            List of violation dicts: {person, zone_name, zone_type}
        """
        violations = []
        with self._lock:
            zones_to_check = (
                {zone_name: self._zones[zone_name]}
                if zone_name and zone_name in self._zones
                else self._zones
            )

        for person in persons:
            for z_name, zone in zones_to_check.items():
                if self.is_in_zone(person.center, z_name):
                    violations.append({
                        "person": person,
                        "zone_name": z_name,
                        "zone_type": zone["type"],
                    })

        return violations

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all zone polygons on the frame.

        Args:
            frame: BGR image to draw on

        Returns:
            Annotated frame
        """
        zone_colors = {
            "restricted": (0, 0, 255),       # Red
            "high_security": (0, 0, 200),    # Dark red
            "entrance": (255, 165, 0),        # Orange
            "outdoor": (0, 255, 255),          # Yellow
        }

        with self._lock:
            for name, zone in self._zones.items():
                color = zone_colors.get(zone["type"], (0, 0, 255))
                polygon = zone["polygon"]

                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], color)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

                cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)

                if len(polygon) > 0:
                    cx = int(np.mean(polygon[:, 0]))
                    cy = int(np.mean(polygon[:, 1]))
                    label = f"{name} ({zone['type']})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
                    cv2.rectangle(frame, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 2), color, -1)
                    cv2.putText(frame, label, (cx, cy), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def get_all_zones(self) -> list[dict]:
        """Get all zone configurations."""
        with self._lock:
            return [
                {"name": z["name"], "points": z["points"], "type": z["type"]}
                for z in self._zones.values()
            ]

    def _save_zones(self):
        """Persist zones to a JSON file."""
        try:
            save_data = {}
            for name, zone in self._zones.items():
                save_data[name] = {
                    "points": zone["points"],
                    "type": zone["type"],
                }
            with open(self._zones_file, "w") as f:
                json.dump(save_data, f, indent=2)
            logger.debug(f"Zones saved to {self._zones_file}")
        except Exception as e:
            logger.error(f"Failed to save zones: {e}")
