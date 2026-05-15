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

        import uuid
        incident_uuid = str(uuid.uuid4())[:8].upper()

        threat_type = threat_event.get("threat_type", "UNKNOWN")
        confidence = threat_event.get("confidence", 0.0)

        # 3. Forensic Overlays
        # Banner with UUID
        banner_text = f" SAFEWATCH FORENSIC | {threat_type} | ID: {incident_uuid} "
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = w / 1200.0
        thickness = max(1, int(w / 600))
        (tw, th), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)
        
        banner_h = th + baseline + 25
        cv2.rectangle(annotated, (0, 0), (w, banner_h), border_color, -1)
        cv2.putText(annotated, banner_text, (15, th + 15), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Confidence Bar
        bar_w = int(w * 0.2)
        bar_h = 10
        bar_x, bar_y = w - bar_w - 20, 15
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + int(bar_w * confidence), bar_y + bar_h), (0, 255, 0), -1)
        cv2.putText(annotated, f"CONF: {confidence:.0%}", (bar_x, bar_y + 30), font, font_scale*0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Bboxes & Labels
        location = threat_event.get("location_bbox")
        if location:
            # Scale bbox if resized
            if frame.shape[1] != w:
                sx, sy = w / frame.shape[1], h / frame.shape[0]
                lx1, ly1, lx2, ly2 = location
                location = (int(lx1*sx), int(ly1*sy), int(lx2*sx), int(ly2*sy))
            
            x1, y1, x2, y2 = location
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), max(1, int(w/400)))
            
            # Label
            label = f"SUBJECT [{threat_type}]"
            (lw, lh), _ = cv2.getTextSize(label, font, font_scale*0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - lh - 10), (x1 + lw + 10, y1), (0, 255, 255), -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), font, font_scale*0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Forensic Watermark & Chain of Custody
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        footer_text = f"EVIDENCE: {camera_name or camera_id} | UTC {dt_str} | SW-SEC-V3"
        cv2.putText(annotated, footer_text, (10, h - 15), font, font_scale*0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Evidence Watermark (Transparent diagonal)
        cv2.putText(annotated, "SECURED FORENSIC EVIDENCE", (int(w*0.1), int(h*0.9)), font, font_scale*1.2, (100, 100, 100), 2, cv2.LINE_AA)

        # 4. Optimized JPEG Encoding
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
