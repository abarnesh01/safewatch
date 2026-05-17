"""
SafeWatch — StreamManager
Manages multiple camera streams with health monitoring and auto-restart.
"""

import time
import threading
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class FramePacket:
    frame: 'np.ndarray'
    camera_id: str
    camera_name: str
    has_motion: bool = True

import numpy as np
from loguru import logger

from capture.camera_stream import CameraStream
from capture.frame_sampler import FrameSampler


class StreamManager:
    """
    Manages multiple CameraStream and FrameSampler instances.
    Provides health monitoring, auto-restart, and centralized frame access.
    """

    def __init__(self, config: dict):
        self._config = config
        self._streams: dict[str, CameraStream] = {}
        self._samplers: dict[str, FrameSampler] = {}
        self._lock = threading.Lock()
        self._health_thread: Optional[threading.Thread] = None
        self._running = False
        self._health_interval = 60
        self._init_streams()
        logger.info(f"StreamManager initialized with {len(self._streams)} cameras")

    def __repr__(self) -> str:
        active = sum(1 for s in self._streams.values() if s.is_connected())
        return f"StreamManager(total={len(self._streams)}, active={active})"

    def _init_streams(self):
        """Initialize camera streams from config."""
        cameras = self._config.get("cameras", [])
        for cam_cfg in cameras:
            if not cam_cfg.get("enabled", False):
                logger.info(f"Camera {cam_cfg['id']} is disabled, skipping")
                continue

            cam_id = cam_cfg["id"]
            resolution = tuple(cam_cfg.get("resolution", [640, 480]))
            fps_target = cam_cfg.get("fps_target", 15)
            frame_skip = cam_cfg.get("frame_skip", 5)

            stream = CameraStream(
                camera_id=cam_id,
                source=cam_cfg["source"],
                resolution=resolution,
                fps_target=fps_target,
                name=cam_cfg.get("name", cam_id),
            )

            sampler = FrameSampler(
                camera_stream=stream,
                frame_skip=frame_skip,
                resolution=resolution,
            )

            with self._lock:
                self._streams[cam_id] = stream
                self._samplers[cam_id] = sampler

            logger.info(f"Camera configured: {cam_id} ({cam_cfg.get('name', cam_id)})")

    def start_all(self):
        """Start all camera streams and the health monitor."""
        self._running = True
        with self._lock:
            for cam_id, stream in self._streams.items():
                stream.start()
                logger.info(f"Started stream: {cam_id}")

        self._health_thread = threading.Thread(
            target=self._health_monitor_loop,
            name="StreamManager-Health",
            daemon=True,
        )
        self._health_thread.start()
        logger.info("All camera streams started")

    def stop_all(self):
        """Stop all camera streams and the health monitor."""
        self._running = False
        with self._lock:
            for cam_id, stream in self._streams.items():
                stream.stop()
                logger.info(f"Stopped stream: {cam_id}")

        if self._health_thread is not None:
            self._health_thread.join(timeout=5.0)
            self._health_thread = None

        logger.info("All camera streams stopped")

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get the latest frame from a specific camera.

        Args:
            camera_id: The camera identifier

        Returns:
            The latest frame, or None if not available.
        """
        with self._lock:
            stream = self._streams.get(camera_id)
        if stream is None:
            return None
        return stream.read()

    def get_latest_frame(self, camera_id: str) -> Optional[FramePacket]:
        """
        Get the latest FramePacket from a specific camera.
        """
        frame = self.get_frame(camera_id)
        if frame is None:
            return None
            
        stream = self.get_stream(camera_id)
        if stream is None:
            return None
            
        return FramePacket(
            frame=frame,
            camera_id=camera_id,
            camera_name=stream.name,
            has_motion=True
        )

    def get_sampler(self, camera_id: str) -> Optional[FrameSampler]:
        """Get the frame sampler for a camera."""
        with self._lock:
            return self._samplers.get(camera_id)

    def get_stream(self, camera_id: str) -> Optional[CameraStream]:
        """Get the camera stream for a camera."""
        with self._lock:
            return self._streams.get(camera_id)

    def get_all_camera_ids(self) -> list[str]:
        """Get list of all camera IDs."""
        with self._lock:
            return list(self._streams.keys())

    def get_status(self) -> dict:
        """
        Get health status of all cameras with advanced intelligence analytics.

        Returns:
            Dict mapping camera_id → status dict
        """
        status = {}
        with self._lock:
            for cam_id, stream in self._streams.items():
                s = stream.get_status()
                # Advanced Intelligence
                s["bandwidth_mbps"] = round((s["fps"] * stream.resolution[0] * stream.resolution[1] * 3) / (1024 * 1024), 2)
                
                # Reliability Score (0.0 - 1.0)
                reconnects = getattr(stream, "_reconnect_count", 0)
                s["reliability_score"] = max(0.0, 1.0 - (reconnects * 0.1))
                
                status[cam_id] = s
        return status

    def _health_monitor_loop(self):
        """Advanced health monitor: scores, frozen detection, and auto-restart."""
        logger.info("Advanced Health Monitor active")
        
        # Track last frame hashes for frozen detection
        last_hashes = defaultdict(list)
        uptime_stats = defaultdict(float)
        
        while self._running:
            time.sleep(self._health_interval)
            if not self._running: break

            with self._lock:
                for cam_id, stream in self._streams.items():
                    status = stream.get_status()
                    score = 100
                    
                    # 1. Connectivity & FPS Check
                    if not status["connected"]:
                        score -= 50
                    elif status["fps"] < (stream.fps_target * 0.7):
                        score -= 20
                    
                    # 2. Frozen Frame Detection
                    frame = stream.read()
                    if frame is not None:
                        # Simple hash for frozen check
                        h = hash(frame.tobytes())
                        last_hashes[cam_id].append(h)
                        if len(last_hashes[cam_id]) > 5:
                            last_hashes[cam_id].pop(0)
                        
                        if len(last_hashes[cam_id]) == 5 and len(set(last_hashes[cam_id])) == 1:
                            logger.warning(f"[{cam_id}] Stream frozen detected!")
                            score -= 60
                    
                    # 3. Buffer Health
                    if status["buffer_size"] > 10:
                        score -= 10

                    # 4. Auto-Recovery Logic
                    if score < 40:
                        logger.error(f"[{cam_id}] Critical failure (Score: {score}). Auto-recovering...")
                        stream.stop()
                        time.sleep(2.0)
                        stream.start()
                        score = 50 # Partial reset after recovery
                    
                    # Store score (this would normally go to DB)
                    status["health_score"] = max(0, score)
                    logger.debug(f"[{cam_id}] Health Score: {status['health_score']}%")

        logger.info("Health monitor stopped")

    def restart_camera(self, camera_id: str) -> bool:
        """
        Restart a specific camera stream.

        Args:
            camera_id: Camera to restart

        Returns:
            True if camera was found and restart initiated
        """
        with self._lock:
            stream = self._streams.get(camera_id)
        if stream is None:
            logger.warning(f"Camera {camera_id} not found")
            return False

        stream.stop()
        time.sleep(1.0)
        stream.start()
        logger.info(f"Camera {camera_id} restarted")
        return True

    def enable_camera(self, camera_id: str) -> bool:
        """Start a previously disabled camera."""
        with self._lock:
            stream = self._streams.get(camera_id)
        if stream is None:
            logger.warning(f"Camera {camera_id} not found in config")
            return False
        if not stream.is_running():
            stream.start()
            logger.info(f"Camera {camera_id} enabled")
        return True

    def disable_camera(self, camera_id: str) -> bool:
        """Stop a running camera."""
        with self._lock:
            stream = self._streams.get(camera_id)
        if stream is None:
            return False
        if stream.is_running():
            stream.stop()
            logger.info(f"Camera {camera_id} disabled")
        return True
