"""
SafeWatch Snapshot Builder
Generates high-quality threat snapshots with overlays and metadata.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from loguru import logger
from threats.fight_detector import ThreatEvent


class SnapshotBuilder:
    """Renders annotated images for alert dispatch."""

    def __init__(self, output_dir: str = "recordings/snapshots") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("SnapshotBuilder initialized")

    def build_snapshot(self, frame: np.ndarray, 
                       event: ThreatEvent, 
                       camera_name: str) -> str:
        """Create a JPEG snapshot with threat banners and metadata."""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. Draw Threat Banner
        banner_h = 60
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (0, 0, 150), -1)
        
        timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        title = f"ALARM: {event.threat_type.upper()} | {event.severity}"
        cv2.putText(annotated, title, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 2. Add Watermark and Camera Info
        cv2.putText(annotated, f"CAM: {camera_name} | {timestamp}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, "SafeWatch AI", (w - 150, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 3. Save File
        filename = f"snapshot_{event.threat_type}_{datetime.now().strftime('%H%M%S_%f')}.jpg"
        filepath = self._output_dir / filename
        
        cv2.imwrite(str(filepath), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        logger.debug("Snapshot saved: {}", filepath)
        return str(filepath)
