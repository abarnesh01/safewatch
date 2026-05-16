"""
SafeWatch — PersonDetector
YOLOv8-based person detection with simple IoU tracking for consistent IDs.
"""

import threading
from typing import Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger


@dataclass
class Person:
    """Represents a detected person with bounding box and tracking info."""
    id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: tuple[int, int]
    area: int
    width: int
    height: int

    def __repr__(self) -> str:
        return (
            f"Person(id={self.id}, confidence={self.confidence:.2f}, "
            f"center={self.center}, area={self.area})"
        )


class PersonDetector:
    """
    YOLOv8-based person detector with IoU-based tracking for consistent person IDs.
    """

    def __init__(self, model_path: str = "models/yolov8n.pt", confidence: float = 0.5, device: str = "cpu"):
        self._model_path = model_path
        self._confidence = confidence
        self._device = device
        self._classes = [0] # Person only
        self._max_tracked = 10
        self._model = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._previous_persons: list[Person] = []
        self._iou_threshold = 0.3
        self._load_model()

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"PersonDetector(model_loaded={loaded}, confidence={self._confidence}, "
            f"tracking={len(self._previous_persons)} persons)"
        )

    def _load_model(self):
        """Load the YOLOv8 model, downloading if necessary."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            self._model.to(self._device)
            logger.info(f"YOLOv8 model loaded from {self._model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            logger.warning("PersonDetector will return empty detections until model is available")
            self._model = None

    def _calculate_iou(self, box1: tuple, box2: tuple) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    def _assign_ids(self, new_detections: list[dict]) -> list[Person]:
        """
        Assign consistent IDs to detections using IoU matching with previous frame.

        Args:
            new_detections: List of raw detection dicts {bbox, confidence}

        Returns:
            List of Person objects with consistent IDs
        """
        persons: list[Person] = []
        used_prev_ids: set[int] = set()

        for det in new_detections:
            bbox = det["bbox"]
            best_iou = 0.0
            best_prev_id = -1

            for prev in self._previous_persons:
                if prev.id in used_prev_ids:
                    continue
                iou = self._calculate_iou(bbox, prev.bbox)
                if iou > best_iou and iou > self._iou_threshold:
                    best_iou = iou
                    best_prev_id = prev.id

            if best_prev_id >= 0:
                person_id = best_prev_id
                used_prev_ids.add(best_prev_id)
            else:
                person_id = self._next_id
                self._next_id += 1

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w = x2 - x1
            h = y2 - y1

            persons.append(Person(
                id=person_id,
                bbox=bbox,
                confidence=det["confidence"],
                center=(cx, cy),
                area=w * h,
                width=w,
                height=h,
            ))

        self._previous_persons = persons
        return persons

    def detect(self, frame: np.ndarray) -> list[Person]:
        """
        Detect persons in a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of Person objects with tracking IDs
        """
        if self._model is None:
            return []

        with self._lock:
            try:
                results = self._model(
                    frame,
                    conf=self._confidence,
                    classes=self._classes,
                    verbose=False,
                )

                raw_detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                        conf = float(boxes.conf[i].cpu().numpy())
                        raw_detections.append({
                            "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                            "confidence": conf,
                        })

                raw_detections = raw_detections[:self._max_tracked]
                persons = self._assign_ids(raw_detections)
                return persons

            except Exception as e:
                logger.error(f"Detection error: {e}")
                return []

    def draw_detections(self, frame: np.ndarray, persons: list[Person]) -> np.ndarray:
        """
        Draw bounding boxes and person IDs on the frame.

        Args:
            frame: BGR image to draw on (modified in place)
            persons: List of Person objects

        Returns:
            The annotated frame
        """
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{person.id} {person.confidence:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
            )

        return frame

    def reset_tracking(self):
        """Reset the person tracking state."""
        with self._lock:
            self._previous_persons = []
            self._next_id = 1
            logger.debug("Person tracking reset")
