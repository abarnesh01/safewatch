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
import threading
from pathlib import Path
from loguru import logger
from utils.runtime_isolation import RuntimePath

from database.db_manager import DatabaseManager
from database.incident_logger import IncidentLogger
from capture.stream_manager import StreamManager
from detection.person_detector import PersonDetector
from detection.pose_estimator import PoseEstimator
from detection.optical_flow import OpticalFlowAnalyzer
from threats.threat_engine import ThreatEngine
from alerts.alert_manager import AlertManager
from alerts.snapshot_builder import SnapshotBuilder
from alerts.telegram_bot import SafeWatchTelegramBot
from utils.observability import ObservabilityEngine


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
        self._stream_manager = StreamManager(self._config)
        
        # 3. Initialize AI Detectors with auto-detected device
        self._person_detector = PersonDetector(
            model_path=self._config["detection"].get("model_type", "yolov8n.pt"),
            confidence=self._config["detection"].get("person_conf", 0.5),
            device=self._device
        )
        self._pose_estimator = PoseEstimator(config=self._config, device=self._device)
        self._flow_analyzers = {}
        self._prev_frames = {}
        
        # 4. Initialize Engines
        self._threat_engine = ThreatEngine(self._config, zone_manager=None) # ZoneManager initialized later
        
        # 5. Initialize Alerting
        self._telegram_bot = None
        if self._config["alerts"].get("telegram_enabled", False):
            self._telegram_bot = SafeWatchTelegramBot(self._config)
        
        self._alert_manager = AlertManager(
            config=self._config,
            telegram_bot=self._telegram_bot,
            incident_logger=self._incident_logger
        )
        
        # 0. Runtime Isolation
        RuntimePath.ensure_isolation()
        
        self._running = True
        self._start_cleanup_service()
        
        # Observability
        self._obs = ObservabilityEngine()
        self._last_fps_time = time.time()
        self._frame_count = 0
        
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
        log_dir = Path(self._config["system"].get("log_dir", "logs"))
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
        self._incident_logger.add_audit_log("SYSTEM", "ENGINE_START", "PROCESS", "SafeWatch surveillance engine initialized")
        

        
        self._stream_manager.start_all()
        
        try:
            while self._running:
                # Process each camera stream
                for cam_id in self._stream_manager.get_all_camera_ids():
                    frame_packet = self._stream_manager.get_latest_frame(cam_id)
                    if not frame_packet:
                        continue
                    
                    # 1. AI Pipeline
                    start_time = time.time()
                    
                    p_start = time.time()
                    persons = self._person_detector.detect(frame_packet.frame)
                    self._obs.record_breakdown("PersonDetection", (time.time() - p_start) * 1000)
                    
                    ps_start = time.time()
                    poses = {}
                    for p in persons:
                        pose = self._pose_estimator.estimate(frame_packet.frame, p.person_id, p.bbox)
                        if pose:
                            poses[p.person_id] = pose
                    self._obs.record_breakdown("PoseEstimation", (time.time() - ps_start) * 1000)
                    
                    f_start = time.time()
                    if cam_id not in self._flow_analyzers:
                        self._flow_analyzers[cam_id] = OpticalFlowAnalyzer(self._config)
                        self._prev_frames[cam_id] = frame_packet.frame
                        
                    prev_frame = self._prev_frames[cam_id]
                    curr_frame = frame_packet.frame
                    flow = self._flow_analyzers[cam_id].analyze(prev_frame, curr_frame)
                    self._prev_frames[cam_id] = curr_frame
                    self._obs.record_breakdown("OpticalFlow", (time.time() - f_start) * 1000)
                    
                    latency = (time.time() - start_time) * 1000
                    
                    # FPS calculation
                    self._frame_count += 1
                    now = time.time()
                    if now - self._last_fps_time >= 1.0:
                        fps = self._frame_count / (now - self._last_fps_time)
                        self._obs.record_fps(fps)
                        self._frame_count = 0
                        self._last_fps_time = now
                    
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
        """Update system telemetry and apply adaptive optimization."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        # Adaptive AI Optimization Layer
        # If CPU > 85%, increase frame skip across all samplers
        if cpu > 85.0:
            logger.warning(f"High CPU detected ({cpu}%). Throttling inference sampling...")
            for sampler in self._stream_manager._samplers.values():
                sampler.update_skip_rate(min(30, sampler._frame_skip + 1))
        elif cpu < 40.0:
            # Gradually recover if idle
            for sampler in self._stream_manager._samplers.values():
                sampler.update_skip_rate(max(5, sampler._frame_skip - 1))

        logger.debug(f"Telemetry [{cam_id}]: Latency={latency:.1f}ms CPU={cpu}% RAM={ram}%")

    def _start_cleanup_service(self) -> None:
        """Background service to clean up stale runtime artifacts."""
        def cleanup_loop():
            while self._running:
                logger.info("Running scheduled runtime cleanup...")
                try:
                    # Clean snapshots older than 7 days
                    retention_days = self._config["system"].get("snapshot_retention_days", 7)
                    threshold = time.time() - (retention_days * 86400)
                    
                    for p in RuntimePath.SNAPSHOTS.glob("*.jpg"):
                        if p.stat().st_mtime < threshold:
                            p.unlink()
                            
                    # Clean telemetry/runtime cache
                    for p in RuntimePath.CACHE.glob("*"):
                        if p.is_file() and p.stat().st_mtime < (time.time() - 3600): # 1 hour
                            p.unlink()
                            
                except Exception as e:
                    logger.error(f"Cleanup service error: {e}")
                
                time.sleep(3600) # Run every hour

        threading.Thread(target=cleanup_loop, name="CleanupService", daemon=True).start()

    def shutdown(self) -> None:
        logger.info("Shutting down SafeWatch...")
        self._incident_logger.add_audit_log("SYSTEM", "ENGINE_SHUTDOWN", "PROCESS", "SafeWatch surveillance engine shutdown")
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
