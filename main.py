"""
SafeWatch — AI-Powered CCTV Threat Detection System
Main entry point for the enterprise production build.
"""

import asyncio
import signal
import sys
import yaml
import time
import psutil
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
    """The core application coordinator with hardware auto-detection."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._config = self._load_config(config_path)
        self._setup_logging()
        
        # Hardware auto-detection
        self._device = self._detect_hardware()
        
        # 1. Initialize Database
        self._db = DatabaseManager(self._config["database"]["path"])
        self._incident_logger = IncidentLogger(self._db)
        
        # 2. Initialize Hardware/Capture
        self._stream_manager = StreamManager()
        
        # 3. Initialize AI Detectors with auto-detected device
        self._person_detector = PersonDetector(
            model_path=self._config["detection"]["yolo"]["model_path"],
            confidence=self._config["detection"]["yolo"]["confidence_threshold"],
            device=self._device
        )
        self._pose_estimator = PoseEstimator(config=self._config, device=self._device)
        self._flow_analyzer = OpticalFlowAnalyzer()
        
        # 4. Initialize Engines
        self._threat_engine = ThreatEngine(self._config, zone_manager=None) # ZoneManager initialized later
        
        # 5. Initialize Alerting
        self._telegram_bot = None
        if self._config["alerts"]["telegram"]["enabled"]:
            from alerts.telegram_bot import SafeWatchTelegramBot
            self._telegram_bot = SafeWatchTelegramBot(
                token=self._config["alerts"]["telegram"]["bot_token"],
                chat_id=self._config["alerts"]["telegram"]["chat_id"]
            )
        
        self._alert_manager = AlertManager(
            config=self._config,
            telegram_bot=self._telegram_bot,
            incident_logger=self._incident_logger
        )
        
        self._running = True
        logger.info(f"SafeWatch App initialized successfully on {self._device.upper()}")

    def _detect_hardware(self) -> str:
        """Detect available hardware acceleration (CUDA, ROCm, etc)."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                logger.info("Hardware Acceleration: NVIDIA CUDA detected")
                return "cuda"
            elif 'ROCMExecutionProvider' in providers:
                logger.info("Hardware Acceleration: AMD ROCm detected")
                return "cuda" # YOLO treats ROCm as cuda usually
            elif 'CoreMLExecutionProvider' in providers:
                logger.info("Hardware Acceleration: Apple CoreML detected")
                return "mps"
        except ImportError:
            pass
        
        logger.info("Hardware Acceleration: CPU only")
        return "cpu"

    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        log_dir = Path(self._config["system"]["log_dir"])
        log_dir.mkdir(exist_ok=True)
        logger.add(log_dir / "safewatch.log", rotation="10 MB", level="INFO")
        
        # Validate Git Worktree
        self._validate_worktree()

    def _validate_worktree(self) -> None:
        """Validate repository hygiene and detect accidental tracking of runtime artifacts."""
        import subprocess
        try:
            # Check for dirty worktree (uncommitted changes)
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, check=False, cwd=str(Path(__file__).parent)
            )
            if status.stdout.strip():
                logger.warning("Repository worktree is DIRTY. Please commit or stash changes before running production.")
                
            # Check for accidental tracking of runtime directories
            # (If these are in git ls-files, they were added by mistake)
            tracked = subprocess.run(
                ["git", "ls-files", "logs/", "snapshots/", "recordings/", "exports/"],
                capture_output=True, text=True, check=False, cwd=str(Path(__file__).parent)
            )
            if tracked.stdout.strip():
                logger.error("DETECTION FAILURE: Runtime artifacts are being tracked by Git!")
                logger.error("Fix: Run 'git rm -r --cached <dir>' and update .gitignore")
                
        except Exception as e:
            logger.debug(f"Git validation skipped: {e}")

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
                    start_time = time.time()
                    persons = self._person_detector.detect(frame_packet.frame)
                    
                    poses = {}
                    for p in persons:
                        pose = self._pose_estimator.estimate(frame_packet.frame, p.person_id, p.bbox)
                        if pose:
                            poses[p.person_id] = pose
                    
                    flow = self._flow_analyzer.analyze(frame_packet.frame)
                    latency = (time.time() - start_time) * 1000
                    
                    # 2. Update Telemetry
                    self._update_telemetry(cam_id, latency)
                    
                    # 3. Threat Detection
                    # ... (rest of the loop)
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

    def _update_telemetry(self, cam_id: str, latency: float):
        """Update system telemetry and share with dashboard."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        # In a production system, this would be pushed to a shared state or DB
        # For this build, we use a simple singleton or log-based approach
        logger.debug(f"Telemetry [{cam_id}]: Latency={latency:.1f}ms CPU={cpu}% RAM={ram}%")

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
