"""
SafeWatch — AI-Powered CCTV Threat Detection System
Main entry point for the enterprise production build.
"""

import asyncio
import signal
import sys
import yaml
from pathlib import Path
from loguru import logger

from database.db_manager import DatabaseManager
from database.incident_logger import IncidentLogger
from capture.stream_manager import StreamManager
from detection.person_detector import PersonDetector
from detection.pose_estimator import PoseEstimator
from detection.optical_flow import OpticalFlowAnalyzer
from threats.threat_engine import ThreatEngine
from alerts.alert_manager import AlertManager
from alerts.snapshot_builder import SnapshotBuilder
from alerts.telegram_bot import TelegramAlertBot


class SafeWatchApp:
    """The core application coordinator."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config = self._load_config(config_path)
        self._setup_logging()
        
        # 1. Initialize Database
        self._db = DatabaseManager(self._config["database"]["path"])
        self._incident_logger = IncidentLogger(self._db)
        
        # 2. Initialize Hardware/Capture
        self._stream_manager = StreamManager()
        
        # 3. Initialize AI Detectors
        self._person_detector = PersonDetector(
            model_path=self._config["detection"]["yolo"]["model_path"],
            confidence=self._config["detection"]["yolo"]["confidence_threshold"]
        )
        self._pose_estimator = PoseEstimator()
        self._flow_analyzer = OpticalFlowAnalyzer()
        
        # 4. Initialize Engines
        self._threat_engine = ThreatEngine(self._config)
        
        # 5. Initialize Alerting
        self._telegram_bot = None
        if self._config["alerts"]["telegram"]["enabled"]:
            self._telegram_bot = TelegramAlertBot(
                token=self._config["alerts"]["telegram"]["bot_token"],
                chat_id=self._config["alerts"]["telegram"]["chat_id"]
            )
        
        self._alert_manager = AlertManager(
            self._telegram_bot, 
            SnapshotBuilder()
        )
        
        self._running = True
        logger.info("SafeWatch App initialized successfully")

    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        log_dir = Path(self._config["system"]["log_dir"])
        log_dir.mkdir(exist_ok=True)
        logger.add(log_dir / "safewatch.log", rotation="10 MB", level="INFO")

    async def run(self) -> None:
        """Main processing loop."""
        logger.info("Starting SafeWatch surveillance engine...")
        
        # Add cameras from config
        for cam_cfg in self._config["cameras"]:
            if cam_cfg["enabled"]:
                self._stream_manager.add_camera(
                    camera_id=cam_cfg["id"],
                    source=cam_cfg["source"],
                    camera_name=cam_cfg["name"]
                )
        
        self._stream_manager.start_all()
        
        try:
            while self._running:
                # Process each camera stream
                for cam_id in self._stream_manager.get_camera_ids():
                    frame_packet = self._stream_manager.get_latest_frame(cam_id)
                    if not frame_packet:
                        continue
                    
                    # 1. AI Pipeline
                    persons = self._person_detector.detect(frame_packet.frame)
                    
                    poses = {}
                    for p in persons:
                        pose = self._pose_estimator.estimate(frame_packet.frame, p.person_id, p.bbox)
                        if pose:
                            poses[p.person_id] = pose
                    
                    flow = self._flow_analyzer.analyze(frame_packet.frame)
                    
                    # 2. Threat Detection
                    events = self._threat_engine.process_frame_data(
                        cam_id, persons, poses, flow
                    )
                    
                    # 3. Alert Handling
                    if events:
                        await self._alert_manager.handle_threats(
                            events, frame_packet.frame, frame_packet.camera_name
                        )
                        # Log to DB
                        for e in events:
                            self._incident_logger.log_incident(
                                camera_id=cam_id,
                                camera_name=frame_packet.camera_name,
                                threat_type=e.threat_type,
                                severity=e.severity,
                                confidence=e.confidence,
                                risk_level=e.severity,
                                description=e.description
                            )

                await asyncio.sleep(0.01) # Load balancing
                
        except Exception as exc:
            logger.exception("Core engine failure: {}", exc)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        logger.info("Shutting down SafeWatch...")
        self._running = False
        self._stream_manager.stop_all()
        self._db.close()
        logger.info("SafeWatch shutdown complete")


if __name__ == "__main__":
    app = SafeWatchApp()
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        pass
