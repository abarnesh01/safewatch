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
        jpeg_quality: int = 85,
    ) -> bytes:
        """
        Build an annotated snapshot image.

        Args:
            frame: BGR image
            threat_event: Threat event dict with threat_type, confidence, severity, etc.
            camera_id: Camera identifier
            timestamp: Unix timestamp
            camera_name: Human-readable camera name

        Returns:
            JPEG image bytes
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        threat_type = threat_event.get("threat_type", "UNKNOWN")
        confidence = threat_event.get("confidence", 0.0)
        severity = threat_event.get("severity", "LOW")

        # Draw colored border based on severity
        border_colors = {
            "LOW": (0, 255, 255),
            "MEDIUM": (0, 165, 255),
            "HIGH": (0, 0, 255),
            "CRITICAL": (255, 0, 128),
        }
        border_color = border_colors.get(severity, (0, 0, 255))
        border_thickness = 6
        cv2.rectangle(
            annotated, (0, 0), (w - 1, h - 1),
            border_color, border_thickness,
        )

        # Draw threat label banner at top
        banner_text = f"  {threat_type} DETECTED — {confidence:.0%} confidence  "
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)

        banner_h = th + baseline + 20
        cv2.rectangle(annotated, (0, 0), (w, banner_h), border_color, -1)
        cv2.putText(
            annotated, banner_text,
            (10, th + 10),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

        # Draw bounding boxes around involved persons
        persons_involved = threat_event.get("persons_involved", [])
        location = threat_event.get("location_bbox")
        if location:
            x1, y1, x2, y2 = location
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if persons_involved:
                ids_text = f"IDs: {', '.join(str(pid) for pid in persons_involved)}"
                cv2.putText(
                    annotated, ids_text,
                    (x1, y1 - 8),
                    font, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
                )

        # Timestamp overlay at bottom right
        dt = datetime.fromtimestamp(timestamp)
        ts_text = dt.strftime("%d/%m/%Y %H:%M:%S")
        (tw2, th2), _ = cv2.getTextSize(ts_text, font, 0.5, 1)
        cv2.rectangle(
            annotated,
            (w - tw2 - 14, h - th2 - 14),
            (w, h),
            (0, 0, 0), -1,
        )
        cv2.putText(
            annotated, ts_text,
            (w - tw2 - 10, h - 8),
            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Camera name at bottom left
        cam_label = camera_name or camera_id
        cv2.rectangle(annotated, (0, h - th2 - 14), (len(cam_label) * 10 + 14, h), (0, 0, 0), -1)
        cv2.putText(
            annotated, cam_label,
            (6, h - 8),
            font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # SafeWatch watermark in top-right corner
        wm_text = "SafeWatch"
        (ww, wh), _ = cv2.getTextSize(wm_text, font, 0.4, 1)
        cv2.putText(
            annotated, wm_text,
            (w - ww - 10, banner_h + wh + 8),
            font, 0.4, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Encode as JPEG
        success, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not success:
            logger.error("Failed to encode snapshot as JPEG")
            return b""

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
