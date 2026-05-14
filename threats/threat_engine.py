"""
SafeWatch Threat Engine
Core intelligence coordinator that executes detectors and aggregates threat events.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from detection.person_detector import DetectedPerson
from detection.pose_estimator import PersonPose, PoseEstimator
from detection.optical_flow import FlowStats, OpticalFlowAnalyzer
from detection.zone_manager import ZoneManager

from classifier.skeleton_analyzer import SkeletonAnalyzer
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


class ThreatEngine:
    """Coordinates the AI pipeline and threat detection logic."""

    RISK_LEVELS = {
        "SAFE": {"color": (0, 255, 0), "score": 0},
        "LOW": {"color": (0, 255, 255), "score": 1},
        "MEDIUM": {"color": (0, 165, 255), "score": 2},
        "HIGH": {"color": (0, 0, 255), "score": 3},
        "CRITICAL": {"color": (255, 0, 255), "score": 4}
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=config.get("system", {}).get("max_workers", 4))
        
        # Initialize Detectors
        self._zone_manager = ZoneManager()
        self._fight_detector = FightDetector()
        self._fall_detector = FallDetector()
        self._harassment_detector = HarassmentDetector()
        self._assault_detector = AssaultDetector()
        self._unconscious_detector = UnconsciousDetector()
        self._trespass_detector = TrespassDetector(self._zone_manager)
        self._crowd_panic_detector = CrowdPanicDetector()
        self._accident_detector = AccidentDetector()
        self._abuse_detector = AbuseDetector()
        
        # Feature Analyzers
        self._skeleton_analyzer = SkeletonAnalyzer()
        self._velocity_tracker = VelocityTracker()
        self._action_classifier = ActionClassifier()
        
        logger.info("ThreatEngine initialized with full detector suite")

    def process_frame_data(self, camera_id: str, 
                           persons: List[DetectedPerson],
                           poses: Dict[int, PersonPose],
                           flow_stats: Optional[FlowStats]) -> List[ThreatEvent]:
        """Execute all threat detectors on processed frame data."""
        
        # 1. Update core analyzers
        features = {}
        velocities = {}
        actions = {}
        
        for pid, pose in poses.items():
            features[pid] = self._skeleton_analyzer.analyze(pose)
            velocities[pid] = self._velocity_tracker.track(pose)
            actions[pid] = self._action_classifier.classify(features[pid], velocities[pid])

        # 2. Parallel Detection
        all_events = []
        
        # Primary Detectors
        all_events.extend(self._fight_detector.detect(camera_id, persons, poses, velocities))
        all_events.extend(self._fall_detector.detect(camera_id, persons, poses, velocities, features))
        all_events.extend(self._harassment_detector.detect(camera_id, persons))
        all_events.extend(self._assault_detector.detect(camera_id, persons, velocities))
        all_events.extend(self._unconscious_detector.detect(camera_id, persons, features, velocities))
        all_events.extend(self._trespass_detector.detect(camera_id, persons))
        all_events.extend(self._crowd_panic_detector.detect(camera_id, flow_stats, len(persons)))
        
        # Meta Detectors (dependent on other events)
        all_events.extend(self._accident_detector.detect(camera_id, all_events))
        all_events.extend(self._abuse_detector.detect(camera_id, all_events))
        
        # 3. Aggregation & Risk Scoring
        return self._aggregate_threats(all_events)

    def _aggregate_threats(self, events: List[ThreatEvent]) -> List[ThreatEvent]:
        """Filter and deduplicate threat events."""
        if not events:
            return []
            
        # Deduplicate by type and person_ids
        seen = set()
        unique_events = []
        for e in events:
            key = (e.threat_type, tuple(sorted(e.person_ids)))
            if key not in seen:
                seen.add(key)
                unique_events.append(e)
                
        return unique_events

    def get_overall_risk(self, events: List[ThreatEvent]) -> str:
        """Determine the highest risk level from active events."""
        if not events:
            return "SAFE"
        
        max_score = 0
        risk_level = "SAFE"
        
        for e in events:
            level = e.severity
            score = self.RISK_LEVELS.get(level, {}).get("score", 0)
            if score > max_score:
                max_score = score
                risk_level = level
                
        return risk_level

    def annotate_frame(self, frame: np.ndarray, 
                       persons: List[DetectedPerson],
                       poses: Dict[int, PersonPose],
                       events: List[ThreatEvent],
                       risk_level: str) -> np.ndarray:
        """Render detection overlays and threat alerts onto the frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Risk Overlay Border
        color = self.RISK_LEVELS.get(risk_level, {}).get("color", (0, 255, 0))
        cv2.rectangle(annotated, (0, 0), (w, h), color, 10)
        
        # Header Banner
        cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 0), -1)
        status_text = f"SafeWatch | RISK: {risk_level} | Alerts: {len(events)}"
        cv2.putText(annotated, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw Person Bounding Boxes
        for p in persons:
            x1, y1, x2, y2 = p.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(annotated, f"ID:{p.person_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw Threat Labels
        y_offset = 100
        for e in events:
            alert_text = f"ALARM: {e.threat_type.upper()} ({e.severity})"
            cv2.putText(annotated, alert_text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30
            
        return annotated
