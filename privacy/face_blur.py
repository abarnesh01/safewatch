import cv2
import numpy as np
from loguru import logger

class PrivacyManager:
    """Handles GDPR/CCPA dynamic face and person blurring based on configured policy."""
    
    def __init__(self, mode: str = "INTERNAL"):
        # Modes: PUBLIC, INTERNAL, SECURITY_ONLY
        self.mode = mode
        logger.info(f"PrivacyManager initialized in {self.mode} mode.")

    def apply_privacy(self, frame: np.ndarray, persons: list) -> np.ndarray:
        """Applies dynamic gaussian blur to bounding boxes based on identity and policy."""
        if self.mode == "SECURITY_ONLY":
            return frame # No redaction for core SOC operators
            
        for p in persons:
            if hasattr(p, 'box') and p.box is not None:
                x1, y1, x2, y2 = map(int, p.box)
                identity = getattr(p, 'identity', {})
                category = identity.get("category", "UNKNOWN")
                
                blur_needed = False
                
                if self.mode == "PUBLIC":
                    # Public facing monitors: Blur everyone
                    blur_needed = True
                elif self.mode == "INTERNAL":
                    # Internal SOC: Blur unknowns, reveal registered employees/threats
                    if category == "UNKNOWN" or category == "Visitor":
                        blur_needed = True
                        
                if blur_needed:
                    # Constrain to frame boundaries
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        # Heavy Gaussian blur to render face unrecognizable
                        roi = cv2.GaussianBlur(roi, (99, 99), 30)
                        frame[y1:y2, x1:x2] = roi
                        
        return frame
