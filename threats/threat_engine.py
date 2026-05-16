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
from detection.zone_manager import ZoneManager
from classifier.velocity_tracker import VelocityTracker
from classifier.action_classifier import ActionClassifier
from utils.observability import ObservabilityEngine
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
        self._classifier = ActionClassifier(config)

        # Modular Detector Registry
        self._detectors = {}
        self._init_detectors()

        # Per-threat cooldown tracking
        self._cooldowns: dict[str, float] = defaultdict(float)
        self._cooldown_seconds = config.get("telegram", {}).get("alert_cooldown_seconds", 30)

        # Temporal smoothing state
        threat_cfg = config.get("threats", {})
        self._smoothing_frames = threat_cfg.get("smoothing_frames", 5)
        self._confirmation_frames = threat_cfg.get("confirmation_frames", 3)
        self._threat_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=self._smoothing_frames))

        # Profiling & Observability
        self._obs = ObservabilityEngine()
        self._profiler: dict[str, list[float]] = defaultdict(list)

        logger.info(
            f"ThreatEngine initialized with temporal smoothing "
            f"(frames={self._smoothing_frames}, confirmation={self._confirmation_frames})"
        )

    def __repr__(self) -> str:
        return f"ThreatEngine(detectors=9, smoothing={self._smoothing_frames})"

    def _init_detectors(self):
        """Initialize detectors into the modular registry."""
        config = self._config
        zone_manager = self._zone_manager # Assumes passed in or None
        
        self._detectors = {
            "fight": FightDetector(config),
            "fall": FallDetector(config),
            "harassment": HarassmentDetector(config),
            "assault": AssaultDetector(config),
            "unconscious": UnconsciousDetector(config),
            "trespass": TrespassDetector(config, zone_manager),
            "crowd_panic": CrowdPanicDetector(config),
            "accident": AccidentDetector(config),
            "abuse": AbuseDetector(config)
        }
        logger.info(f"ThreatEngine Registry initialized with {len(self._detectors)} modular detectors.")

    def _get_detector(self, name: str):
        return self._detectors.get(name)

    def analyze(self, frame_data: dict) -> ThreatReport:
        """
        Analyze a frame for threats with temporal confidence smoothing.
        """
        frame = frame_data.get("frame")
        camera_id = frame_data.get("camera_id", "UNKNOWN")
        timestamp = frame_data.get("timestamp", time.time())
        persons = frame_data.get("persons", [])
        poses = frame_data.get("poses", [])
        flow_result = frame_data.get("flow_result")
        velocity_tracker = frame_data.get("velocity_tracker")

        raw_threats: list[ThreatEvent] = []

        # Run detectors in parallel using thread pool
        futures = {}

        if persons and poses and velocity_tracker:
            # Profiling logic
            def profiled_run(name, detector_func, *args):
                start = time.time()
                try:
                    res = detector_func(*args)
                    lat = (time.time() - start) * 1000
                    self._obs.record_latency(name, lat)
                    with self._lock:
                        self._profiler[name].append(lat)
                        if len(self._profiler[name]) > 100: self._profiler[name].pop(0)
                    return res
                except Exception as e:
                    self._obs.record_error(name)
                    logger.error(f"Detector '{name}' execution trace failure: {e}")
                    raise

            for name in ["fight", "fall", "harassment", "assault", "unconscious", "abuse"]:
                detector = self._get_detector(name)
                if detector:
                    futures[name] = self._executor.submit(
                        profiled_run, name, detector.detect, persons, poses, velocity_tracker
                    )

        if persons:
            trespass_detector = self._get_detector("trespass")
            if trespass_detector:
                futures["trespass"] = self._executor.submit(
                    trespass_detector.detect, persons
                )

        fall_events = []
        for name, future in futures.items():
            try:
                result = future.result(timeout=2.0)
                if name == "fall":
                    fall_events = result
                raw_threats.extend(result)
            except Exception as e:
                logger.error(f"Detector '{name}' failed: {e}")

        if flow_result is not None and velocity_tracker is not None:
            try:
                crowd_detector = self._get_detector("crowd_panic")
                if crowd_detector:
                    raw_threats.extend(crowd_detector.detect(
                        persons, flow_result, velocity_tracker, fall_events
                    ))
            except Exception as e:
                logger.error(f"CrowdPanicDetector failed: {e}")

        if velocity_tracker is not None:
            try:
                accident_detector = self._get_detector("accident")
                if accident_detector:
                    raw_threats.extend(accident_detector.detect(
                        persons, poses, flow_result, velocity_tracker, fall_events
                    ))
            except Exception as e:
                logger.error(f"AccidentDetector failed: {e}")

        # 1. Apply temporal smoothing and confirmation
        stabilized_threats = self._stabilize_threats(raw_threats, camera_id)

        # 2. Secondary AI Verification Layer
        verified_threats = self._verify_threats(stabilized_threats, frame_data)

        # 3. Multi-Threat Correlation Engine
        correlated_threats = self._correlate_threats(verified_threats)

        # 3. Apply global confidence filter
        global_min_conf = self._config.get("threats", {}).get("global_min_confidence", 0.0)
        filtered_threats = [t for t in correlated_threats if t.confidence >= global_min_conf]

        # 4. Apply cooldown filtering
        alertable_threats = self._apply_cooldowns(filtered_threats, camera_id, timestamp)

        # Set timestamps
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        for threat in alertable_threats:
            threat.timestamp = ts_str

        risk_level = self.get_risk_level(alertable_threats)

        annotated = None
        if frame is not None:
            annotated = self._draw_threat_overlays(
                frame.copy(), alertable_threats, risk_level
            )

        return ThreatReport(
            camera_id=camera_id,
            timestamp=timestamp,
            threats_detected=alertable_threats,
            annotated_frame=annotated,
            overall_risk_level=risk_level,
        )

    def _correlate_threats(self, threats: list[ThreatEvent]) -> list[ThreatEvent]:
        """Correlate and group related threats into unified incidents."""
        if not threats: return []
        
        correlated = []
        threat_types = {t.threat_type for t in threats}
        
        # Rule 1: Escalation Mapping (e.g., ASSAULT -> UNCONSCIOUS)
        escalations = {
            "ASSAULT": ["FALL", "UNCONSCIOUS"],
            "FIGHT": ["ASSAULT", "ACCIDENT"],
            "HARASSMENT": ["ASSAULT", "ABUSE"]
        }
        
        processed = set()
        for threat in threats:
            if threat.threat_type in processed: continue
            
            # Look for related escalations
            related = escalations.get(threat.threat_type, [])
            found_related = [t for t in threats if t.threat_type in related and t.threat_type not in processed]
            
            if found_related:
                # Group into a unified incident
                threat.description = f"{threat.threat_type} escalating to {', '.join(t.threat_type for t in found_related)}"
                threat.severity = "CRITICAL"
                threat.confidence = max(threat.confidence, max(t.confidence for t in found_related))
                for t in found_related: processed.add(t.threat_type)
            
            correlated.append(threat)
            processed.add(threat.threat_type)
            
        return correlated

    def _verify_threats(self, threats: list[ThreatEvent], frame_data: dict) -> list[ThreatEvent]:
        """Perform secondary AI verification on rule-based triggers."""
        if not threats or not self._classifier: return threats
        
        verified = []
        persons = frame_data.get("persons", [])
        poses = frame_data.get("poses", [])
        
        for threat in threats:
            # Only verify specific physical threats
            if threat.threat_type not in ["FIGHT", "FALL", "ASSAULT", "ABUSE"]:
                verified.append(threat)
                continue
                
            # Run AI classification for involved persons
            max_ai_conf = 0.0
            behavior_boost = 0.0
            
            for pid in threat.persons_involved:
                # Find the person object
                person = next((p for p in persons if p.id == pid), None)
                if person:
                    res = self._classifier.classify(person, poses)
                    # Fusion: combine rule confidence with AI confidence
                    if res.action_class == threat.threat_type:
                        max_ai_conf = max(max_ai_conf, res.confidence)
                        behavior_boost = max(behavior_boost, getattr(res, "behavior_score", 0.0))
            
            # Confidence Fusion Logic
            # If AI confirms, boost confidence. If AI strongly disagrees, penalize.
            if max_ai_conf > 0.4:
                threat.confidence = (threat.confidence * 0.4) + (max_ai_conf * 0.4) + (behavior_boost * 0.2)
                verified.append(threat)
            elif threat.severity == "CRITICAL":
                # Critical threats are preserved but marked for audit
                threat.description += " (AI Verification Pending)"
                verified.append(threat)
            else:
                logger.debug(f"AI Verification rejected {threat.threat_type} (AI Conf: {max_ai_conf:.2f})")
                
        return verified

    def _stabilize_threats(self, threats: list[ThreatEvent], camera_id: str) -> list[ThreatEvent]:
        """Apply rolling averaging and multi-frame confirmation."""
        stabilized = []
        current_types = {t.threat_type for t in threats}

        with self._lock:
            # Update history for each threat type
            for threat in threats:
                key = f"{camera_id}:{threat.threat_type}"
                self._threat_history[key].append(threat.confidence)

            # Check all tracked threats
            active_keys = [k for k in self._threat_history if k.startswith(f"{camera_id}:")]
            for key in active_keys:
                threat_type = key.split(":")[1]
                history = self._threat_history[key]
                
                if len(history) < self._confirmation_frames:
                    continue
                
                # Multi-frame confirmation: must have enough recent detections
                recent_detections = sum(1 for conf in list(history)[-self._confirmation_frames:] if conf > 0)
                if recent_detections < self._confirmation_frames:
                    # Decay if not detected in latest frame
                    if threat_type not in current_types:
                        history.append(0.0)
                    continue

                # Rolling confidence average
                avg_confidence = sum(history) / len(history)
                
                if avg_confidence > 0:
                    # Find the original threat object to preserve metadata
                    original = next((t for t in threats if t.threat_type == threat_type), None)
                    if original:
                        original.confidence = round(avg_confidence, 3)
                        stabilized.append(original)

        return stabilized

    def _apply_cooldowns(
        self,
        threats: list[ThreatEvent],
        camera_id: str,
        timestamp: float,
    ) -> list[ThreatEvent]:
        """Filter threats that are within cooldown period, while allowing CRITICAL bypass."""
        filtered = []
        with self._lock:
            for threat in threats:
                key = f"{camera_id}:{threat.threat_type}"
                last_time = self._cooldowns.get(key, 0)
                
                # Critical threats can bypass or have reduced cooldown
                cooldown = self._cooldown_seconds
                if threat.severity == "CRITICAL":
                    cooldown = 5.0 # Fast response for critical incidents

                if timestamp - last_time >= cooldown:
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
