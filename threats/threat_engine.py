"""
SafeWatch — ThreatEngine
Central coordinator that runs all threat detectors and aggregates results.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger

from detection.person_detector import Person
from detection.pose_estimator import PoseResult
from detection.optical_flow import FlowResult
from detection.zone_manager import ZoneManager
from classifier.velocity_tracker import VelocityTracker
from classifier.action_classifier import ActionClassifier
from threats.fight_detector import FightDetector, ThreatEvent
from threats.fall_detector import FallDetector
from threats.harassment_detector import HarassmentDetector
from threats.assault_detector import AssaultDetector
from threats.unconscious_detector import UnconsciousDetector
from threats.trespass_detector import TrespassDetector
from threats.crowd_panic_detector import CrowdPanicDetector
from threats.accident_detector import AccidentDetector
from threats.abuse_detector import AbuseDetector


@dataclass
class ThreatReport:
    """Aggregated threat analysis report for a single frame."""
    camera_id: str
    timestamp: float
    threats_detected: list[ThreatEvent]
    annotated_frame: Optional[np.ndarray]
    overall_risk_level: str

    def __repr__(self) -> str:
        return (
            f"ThreatReport(camera='{self.camera_id}', "
            f"threats={len(self.threats_detected)}, "
            f"risk='{self.overall_risk_level}')"
        )


RISK_COLORS = {
    "SAFE": (0, 200, 0),       # Green
    "LOW": (0, 255, 255),      # Yellow
    "MEDIUM": (0, 165, 255),   # Orange
    "HIGH": (0, 0, 255),       # Red
    "CRITICAL": (255, 0, 128), # Purple
}


class ThreatEngine:
    """
    Central threat analysis coordinator.
    Instantiates all detectors and runs them in parallel via ThreadPoolExecutor.
    """

    def __init__(self, config: dict, zone_manager: ZoneManager):
        self._config = config
        self._zone_manager = zone_manager
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Instantiate all detectors
        self._fight_detector = FightDetector(config)
        self._fall_detector = FallDetector(config)
        self._harassment_detector = HarassmentDetector(config)
        self._assault_detector = AssaultDetector(config)
        self._unconscious_detector = UnconsciousDetector(config)
        self._trespass_detector = TrespassDetector(config, zone_manager)
        self._crowd_panic_detector = CrowdPanicDetector(config)
        self._accident_detector = AccidentDetector(config)
        self._abuse_detector = AbuseDetector(config)

        # Per-threat cooldown tracking
        self._cooldowns: dict[str, float] = defaultdict(float)
        self._cooldown_seconds = config.get("telegram", {}).get("alert_cooldown_seconds", 30)

        logger.info("ThreatEngine initialized with all detectors")

    def __repr__(self) -> str:
        return "ThreatEngine(detectors=9)"

    def analyze(self, frame_data: dict) -> ThreatReport:
        """
        Analyze a frame for threats using all enabled detectors.

        Args:
            frame_data: Dict with keys:
                - frame: np.ndarray (BGR image)
                - camera_id: str
                - timestamp: float
                - persons: list[Person]
                - poses: list[PoseResult]
                - flow_result: FlowResult
                - zones: ZoneManager (optional)
                - velocity_tracker: VelocityTracker

        Returns:
            ThreatReport with all detected threats and annotated frame
        """
        frame = frame_data.get("frame")
        camera_id = frame_data.get("camera_id", "UNKNOWN")
        timestamp = frame_data.get("timestamp", time.time())
        persons = frame_data.get("persons", [])
        poses = frame_data.get("poses", [])
        flow_result = frame_data.get("flow_result")
        velocity_tracker = frame_data.get("velocity_tracker")

        all_threats: list[ThreatEvent] = []

        # Run detectors in parallel using thread pool
        futures = {}

        # Fight, fall, harassment, assault, unconscious, abuse — need persons+poses+velocity
        if persons and poses and velocity_tracker:
            futures["fight"] = self._executor.submit(
                self._fight_detector.detect, persons, poses, velocity_tracker
            )
            futures["fall"] = self._executor.submit(
                self._fall_detector.detect, persons, poses, velocity_tracker
            )
            futures["harassment"] = self._executor.submit(
                self._harassment_detector.detect, persons, poses, velocity_tracker
            )
            futures["assault"] = self._executor.submit(
                self._assault_detector.detect, persons, poses, velocity_tracker
            )
            futures["unconscious"] = self._executor.submit(
                self._unconscious_detector.detect, persons, poses, velocity_tracker
            )
            futures["abuse"] = self._executor.submit(
                self._abuse_detector.detect, persons, poses, velocity_tracker
            )

        # Trespass — needs persons + zones
        if persons:
            futures["trespass"] = self._executor.submit(
                self._trespass_detector.detect, persons
            )

        # Collect results, especially fall events for cross-detector use
        fall_events = []
        detector_results: dict[str, list] = {}
        
        global_min_conf = self._config.get("threats", {}).get("global_min_confidence", 0.0)

        for name, future in futures.items():
            try:
                result = future.result(timeout=2.0)
                # Apply global confidence filter
                result = [t for t in result if t.confidence >= global_min_conf]
                
                detector_results[name] = result
                if name == "fall":
                    fall_events = result
                all_threats.extend(result)
            except Exception as e:
                logger.error(f"Detector '{name}' failed: {e}")
                detector_results[name] = []

        # Crowd panic and accident need fall_events from above
        if flow_result is not None and velocity_tracker is not None:
            try:
                panic_events = self._crowd_panic_detector.detect(
                    persons, flow_result, velocity_tracker, fall_events
                )
                panic_events = [t for t in panic_events if t.confidence >= global_min_conf]
                all_threats.extend(panic_events)
            except Exception as e:
                logger.error(f"CrowdPanicDetector failed: {e}")

        if velocity_tracker is not None:
            try:
                accident_events = self._accident_detector.detect(
                    persons, poses, flow_result, velocity_tracker, fall_events
                )
                accident_events = [t for t in accident_events if t.confidence >= global_min_conf]
                all_threats.extend(accident_events)
            except Exception as e:
                logger.error(f"AccidentDetector failed: {e}")

        # Apply cooldown filtering
        filtered_threats = self._apply_cooldowns(all_threats, camera_id, timestamp)

        # Set timestamps on threats
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        for threat in filtered_threats:
            threat.timestamp = ts_str

        # Calculate overall risk level
        risk_level = self.get_risk_level(filtered_threats)

        # Draw threat overlays on frame
        annotated = None
        if frame is not None:
            annotated = self._draw_threat_overlays(
                frame.copy(), filtered_threats, risk_level
            )

        return ThreatReport(
            camera_id=camera_id,
            timestamp=timestamp,
            threats_detected=filtered_threats,
            annotated_frame=annotated,
            overall_risk_level=risk_level,
        )

    def _apply_cooldowns(
        self,
        threats: list[ThreatEvent],
        camera_id: str,
        timestamp: float,
    ) -> list[ThreatEvent]:
        """Filter threats that are within cooldown period."""
        filtered = []
        with self._lock:
            for threat in threats:
                key = f"{camera_id}:{threat.threat_type}"
                last_time = self._cooldowns.get(key, 0)
                if timestamp - last_time >= self._cooldown_seconds:
                    filtered.append(threat)
                    self._cooldowns[key] = timestamp
        return filtered

    def get_risk_level(self, threats: list[ThreatEvent]) -> str:
        """
        Calculate overall risk level from detected threats.

        Returns:
            One of: SAFE, LOW, MEDIUM, HIGH, CRITICAL
        """
        if not threats:
            return "SAFE"

        severity_scores = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "CRITICAL": 4,
        }

        max_severity = 0
        for threat in threats:
            s = severity_scores.get(threat.severity, 0)
            max_severity = max(max_severity, s)

        if len(threats) >= 3:
            max_severity = min(4, max_severity + 1)

        risk_map = {0: "SAFE", 1: "LOW", 2: "MEDIUM", 3: "HIGH", 4: "CRITICAL"}
        return risk_map.get(max_severity, "SAFE")

    def _draw_threat_overlays(
        self,
        frame: np.ndarray,
        threats: list[ThreatEvent],
        risk_level: str,
    ) -> np.ndarray:
        """Draw threat overlays with colored borders and labels."""
        h, w = frame.shape[:2]
        border_color = RISK_COLORS.get(risk_level, (0, 200, 0))
        border_thickness = 4

        cv2.rectangle(
            frame, (0, 0), (w - 1, h - 1),
            border_color, border_thickness,
        )

        risk_label = f"RISK: {risk_level}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (tw, th), _ = cv2.getTextSize(risk_label, font, font_scale, thickness)

        cv2.rectangle(frame, (w - tw - 14, 2), (w - 2, th + 12), border_color, -1)
        cv2.putText(
            frame, risk_label,
            (w - tw - 10, th + 6),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

        for i, threat in enumerate(threats):
            severity_color = {
                "LOW": (0, 255, 255),
                "MEDIUM": (0, 165, 255),
                "HIGH": (0, 0, 255),
                "CRITICAL": (255, 0, 128),
            }.get(threat.severity, (0, 255, 255))

            x1, y1, x2, y2 = threat.location_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), severity_color, 2)

            label = f"{threat.threat_type} {threat.confidence:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, font, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), severity_color, -1)
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - 3),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return frame

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
        logger.info("ThreatEngine shutdown")
