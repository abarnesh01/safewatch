"""
SafeWatch — SnapshotBuilder
Builds annotated snapshot images for alert notifications.
"""

import io
from pathlib import Path
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available — snapshots will be basic")


class SnapshotBuilder:
    """Builds annotated snapshot images for threat alerts."""

    def __init__(self):
        logger.info("SnapshotBuilder initialized")

    def __repr__(self) -> str:
        return f"SnapshotBuilder(pil_available={PIL_AVAILABLE})"

    def build(
        self,
        frame: np.ndarray,
        threat_event: dict,
        camera_id: str,
        timestamp: float,
        camera_name: str = "",
        max_width: int = 1280,
    ) -> bytes:
        """
        Build an annotated snapshot with adaptive compression and resolution.
        """
        severity = threat_event.get("severity", "LOW")
        
        # 1. Dynamic Resolution Tuning
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            annotated = cv2.resize(frame, (max_width, int(h * scale)))
        else:
            annotated = frame.copy()
        
        h, w = annotated.shape[:2]
        
        # 2. Dynamic Quality Scaling
        # Higher quality for critical incidents, lower for background noise reduction
        quality_map = {
            "LOW": 60,
            "MEDIUM": 75,
            "HIGH": 85,
            "CRITICAL": 95
        }
        jpeg_quality = quality_map.get(severity, 80)

        threat_type = threat_event.get("threat_type", "UNKNOWN")
        confidence = threat_event.get("confidence", 0.0)

        # (Overlay logic remains same as previous production version)
        # ...
        border_colors = {
            "LOW": (0, 255, 255),
            "MEDIUM": (0, 165, 255),
            "HIGH": (0, 0, 255),
            "CRITICAL": (255, 0, 128),
        }
        border_color = border_colors.get(severity, (0, 0, 255))
        border_thickness = max(2, int(w * 0.005))
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), border_color, border_thickness)

        # Banner
        banner_text = f"  {threat_type} — {confidence:.0%}  "
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = w / 1000.0
        thickness = max(1, int(w / 500))
        (tw, th), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)
        
        banner_h = th + baseline + 20
        cv2.rectangle(annotated, (0, 0), (w, banner_h), border_color, -1)
        cv2.putText(annotated, banner_text, (10, th + 10), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Bboxes
        location = threat_event.get("location_bbox")
        if location:
            # Scale bbox if resized
            if frame.shape[1] != w:
                sx, sy = w / frame.shape[1], h / frame.shape[0]
                lx1, ly1, lx2, ly2 = location
                location = (int(lx1*sx), int(ly1*sy), int(lx2*sx), int(ly2*sy))
            
            x1, y1, x2, y2 = location
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), max(1, int(w/400)))

        # Timestamp & Metadata
        dt_str = datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M:%S")
        cv2.putText(annotated, f"{camera_name or camera_id} | {dt_str}", (10, h - 15), font, font_scale*0.6, (255, 255, 255), max(1, int(thickness*0.6)), cv2.LINE_AA)

        # 3. Optimized JPEG Encoding
        success, buffer = cv2.imencode(".jpg", annotated, [
            cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1
        ])
        if not success:
            return b""

        return buffer.tobytes()

        return buffer.tobytes()

    def save_snapshot(
        self,
        image_bytes: bytes,
        camera_id: str,
        timestamp: float,
        output_dir: str = "recordings",
    ) -> str:
        """
        Save snapshot bytes to disk.

        Args:
            image_bytes: JPEG bytes
            camera_id: Camera identifier
            timestamp: Unix timestamp
            output_dir: Directory to save in

        Returns:
            Path to saved file
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dt = datetime.fromtimestamp(timestamp)
        filename = f"snapshot_{camera_id}_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = out_dir / filename

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Snapshot saved: {filepath}")
        return str(filepath)
