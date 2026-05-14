"""
SafeWatch Stream Manager
Centralized multi-camera management with health monitoring and auto-recovery.
"""

import time
import threading
from typing import Optional

from loguru import logger

from capture.camera_stream import CameraStream, FramePacket


class StreamManager:
    """Manages multiple camera streams with health monitoring."""

    def __init__(self, health_check_interval: int = 30) -> None:
        self._streams: dict[str, CameraStream] = {}
        self._health_interval = health_check_interval
        self._running = threading.Event()
        self._health_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        logger.info("StreamManager initialized")

    def add_camera(self, camera_id: str, source, camera_name: str = "",
                   fps_target: int = 15, resolution: tuple = (1280, 720),
                   queue_size: int = 128, reconnect_delay: int = 5,
                   max_reconnect: int = 50) -> None:
        with self._lock:
            if camera_id in self._streams:
                logger.warning("Camera {} already registered", camera_id)
                return
            stream = CameraStream(
                camera_id=camera_id, source=source,
                camera_name=camera_name, fps_target=fps_target,
                resolution=tuple(resolution), queue_size=queue_size,
                reconnect_delay=reconnect_delay, max_reconnect=max_reconnect,
            )
            self._streams[camera_id] = stream
            logger.info("Camera {} added: {}", camera_id, camera_name)

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            stream = self._streams.pop(camera_id, None)
            if stream:
                stream.stop()
                logger.info("Camera {} removed", camera_id)

    def start_all(self) -> None:
        self._running.set()
        with self._lock:
            for cam_id, stream in self._streams.items():
                if not stream.is_running:
                    stream.start()
                    logger.info("Started camera {}", cam_id)
        self._health_thread = threading.Thread(
            target=self._health_monitor, name="health-monitor", daemon=True,
        )
        self._health_thread.start()
        logger.info("All cameras started ({} total)", len(self._streams))

    def stop_all(self) -> None:
        self._running.clear()
        with self._lock:
            for cam_id, stream in self._streams.items():
                stream.stop()
                logger.info("Stopped camera {}", cam_id)
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5)
        logger.info("All cameras stopped")

    def get_frame(self, camera_id: str, timeout: float = 1.0) -> Optional[FramePacket]:
        stream = self._streams.get(camera_id)
        if stream:
            return stream.get_frame(timeout=timeout)
        return None

    def get_latest_frame(self, camera_id: str) -> Optional[FramePacket]:
        stream = self._streams.get(camera_id)
        if stream:
            return stream.get_latest_frame()
        return None

    def get_all_latest_frames(self) -> dict[str, Optional[FramePacket]]:
        frames = {}
        with self._lock:
            for cam_id, stream in self._streams.items():
                frames[cam_id] = stream.get_latest_frame()
        return frames

    def get_camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._streams.keys())

    def get_health_all(self) -> dict[str, dict]:
        with self._lock:
            return {cid: s.get_health() for cid, s in self._streams.items()}

    def get_camera_count(self) -> int:
        return len(self._streams)

    def _health_monitor(self) -> None:
        while self._running.is_set():
            time.sleep(self._health_interval)
            if not self._running.is_set():
                break
            with self._lock:
                for cam_id, stream in self._streams.items():
                    health = stream.get_health()
                    if not health["connected"] and health["running"]:
                        logger.warning("Camera {} disconnected, will auto-reconnect", cam_id)
                    if health["error_count"] > 100:
                        logger.error("Camera {} excessive errors: {}", cam_id, health["error_count"])
