"""
SafeWatch Person Detector
YOLOv8-based person detection with tracking and CPU optimization.
"""

import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not installed, person detection disabled")


@dataclass
class DetectedPerson:
    """Detected person bounding box with metadata."""
    person_id: int = 0
    bbox: tuple = (0, 0, 0, 0)  # x1, y1, x2, y2
    confidence: float = 0.0
    center: tuple = (0, 0)
    width: int = 0
    height: int = 0
    area: int = 0


class PersonDetector:
    """YOLOv8 person detector optimized for CPU inference."""

    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.45, iou_threshold: float = 0.5,
                 img_size: int = 640, device: str = "cpu") -> None:
        self._model_path = Path(model_path)
        self._confidence = confidence
        self._iou = iou_threshold
        self._img_size = img_size
        self._device = device
        self._model: Optional[YOLO] = None
        self._initialized = False
        self._load_model()

    def _load_model(self) -> None:
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available, detector will not function")
            return
        try:
            self._model = YOLO(str(self._model_path))
            self._model.to(self._device)
            self._initialized = True
            logger.info("YOLOv8 model loaded: {} (device={})", self._model_path, self._device)
        except Exception as exc:
            logger.error("Failed to load YOLO model: {}", exc)
            self._initialized = False

    def detect(self, frame: np.ndarray, track: bool = True) -> list[DetectedPerson]:
        if not self._initialized or self._model is None:
            return []
        try:
            if track:
                results = self._model.track(
                    frame, persist=True, conf=self._confidence,
                    iou=self._iou, imgsz=self._img_size,
                    device=self._device, classes=[self.PERSON_CLASS_ID],
                    verbose=False,
                )
            else:
                results = self._model.predict(
                    frame, conf=self._confidence, iou=self._iou,
                    imgsz=self._img_size, device=self._device,
                    classes=[self.PERSON_CLASS_ID], verbose=False,
                )

            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    if cls_id != self.PERSON_CLASS_ID:
                        continue
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    pid = 0
                    if boxes.id is not None:
                        pid = int(boxes.id[i].item())
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w // 2
                    cy = y1 + h // 2
                    persons.append(DetectedPerson(
                        person_id=pid, bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf, center=(cx, cy), width=w, height=h, area=w * h,
                    ))
            return persons
        except Exception as exc:
            logger.error("Person detection error: {}", exc)
            return []

    def draw_detections(self, frame: np.ndarray, persons: list[DetectedPerson],
                        color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        import cv2
        annotated = frame.copy()
        for p in persons:
            x1, y1, x2, y2 = p.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{p.person_id} {p.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return annotated

    @property
    def is_ready(self) -> bool:
        return self._initialized
