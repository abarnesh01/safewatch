"""
SafeWatch Camera Stream
Thread-safe RTSP/USB camera capture with auto-reconnect and FPS tracking.
"""

import time
import threading
from queue import Queue, Full
from typing import Optional
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger


@dataclass
class FramePacket:
    """Container for a captured frame with metadata."""
    frame: np.ndarray = field(repr=False)
    camera_id: str = ""
    camera_name: str = ""
    timestamp: float = 0.0
    frame_number: int = 0
    width: int = 0
    height: int = 0


class CameraStream:
    """Threaded camera stream with auto-reconnect and health monitoring."""

    def __init__(self, camera_id: str, source, camera_name: str = "",
                 fps_target: int = 15, resolution: tuple = (1280, 720),
                 queue_size: int = 128, reconnect_delay: int = 5,
                 max_reconnect: int = 50) -> None:
        self._camera_id = camera_id
        self._source = source
        self._camera_name = camera_name or camera_id
        self._fps_target = fps_target
        self._resolution = resolution
        self._queue_size = queue_size
        self._reconnect_delay = reconnect_delay
        self._max_reconnect = max_reconnect

        self._frame_queue: Queue = Queue(maxsize=queue_size)
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._connected = threading.Event()
        self._lock = threading.Lock()

        self._frame_count = 0
        self._error_count = 0
        self._fps_actual = 0.0
        self._last_frame_time = 0.0
        self._start_time = 0.0
        self._fps_samples: list = []

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def camera_name(self) -> str:
        return self._camera_name

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def uptime(self) -> float:
        if self._start_time > 0:
            return time.time() - self._start_time
        return 0.0

    @property
    def queue_size(self) -> int:
        return self._frame_queue.qsize()

    def _connect(self) -> bool:
        with self._lock:
            try:
                if self._capture is not None:
                    self._capture.release()

                src = self._source
                if isinstance(src, str) and src.isdigit():
                    src = int(src)

                self._capture = cv2.VideoCapture(src)

                if isinstance(src, int):
                    self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
                    self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
                    self._capture.set(cv2.CAP_PROP_FPS, self._fps_target)

                if isinstance(src, str) and src.startswith("rtsp"):
                    self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)

                if self._capture.isOpened():
                    self._connected.set()
                    logger.info("Camera {} connected: {}", self._camera_id, src)
                    return True

                logger.warning("Camera {} failed to open: {}", self._camera_id, src)
                self._connected.clear()
                return False

            except Exception as exc:
                logger.error("Camera {} connection error: {}", self._camera_id, exc)
                self._error_count += 1
                self._connected.clear()
                return False

    def _reconnect_loop(self) -> bool:
        for attempt in range(1, self._max_reconnect + 1):
            if not self._running.is_set():
                return False
            logger.info("Camera {} reconnect attempt {}/{}",
                        self._camera_id, attempt, self._max_reconnect)
            if self._connect():
                return True
            time.sleep(self._reconnect_delay)
        logger.error("Camera {} max reconnect attempts exceeded", self._camera_id)
        return False

    def _update_fps(self) -> None:
        now = time.time()
        if self._last_frame_time > 0:
            dt = now - self._last_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self._fps_samples.append(instant_fps)
                if len(self._fps_samples) > 30:
                    self._fps_samples.pop(0)
                self._fps_actual = sum(self._fps_samples) / len(self._fps_samples)
        self._last_frame_time = now

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / max(self._fps_target, 1)
        self._start_time = time.time()

        if not self._connect():
            if not self._reconnect_loop():
                return

        while self._running.is_set():
            loop_start = time.time()

            if not self._connected.is_set():
                if not self._reconnect_loop():
                    break
                continue

            try:
                ret, frame = self._capture.read()
                if not ret or frame is None:
                    self._error_count += 1
                    self._connected.clear()
                    logger.warning("Camera {} frame read failed", self._camera_id)
                    continue

                self._frame_count += 1
                self._update_fps()

                h, w = frame.shape[:2]
                packet = FramePacket(
                    frame=frame,
                    camera_id=self._camera_id,
                    camera_name=self._camera_name,
                    timestamp=time.time(),
                    frame_number=self._frame_count,
                    width=w,
                    height=h,
                )

                try:
                    self._frame_queue.put_nowait(packet)
                except Full:
                    try:
                        self._frame_queue.get_nowait()
                    except Exception:
                        pass
                    try:
                        self._frame_queue.put_nowait(packet)
                    except Full:
                        pass

                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as exc:
                self._error_count += 1
                logger.error("Camera {} capture error: {}", self._camera_id, exc)
                self._connected.clear()
                time.sleep(0.5)

    def start(self) -> None:
        if self._running.is_set():
            logger.warning("Camera {} already running", self._camera_id)
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"cam-{self._camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Camera {} stream started", self._camera_id)

    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None
        self._connected.clear()
        logger.info("Camera {} stream stopped", self._camera_id)

    def get_frame(self, timeout: float = 1.0) -> Optional[FramePacket]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except Exception:
            return None

    def get_latest_frame(self) -> Optional[FramePacket]:
        packet = None
        while not self._frame_queue.empty():
            try:
                packet = self._frame_queue.get_nowait()
            except Exception:
                break
        return packet

    def get_health(self) -> dict:
        return {
            "camera_id": self._camera_id,
            "camera_name": self._camera_name,
            "connected": self.is_connected,
            "running": self.is_running,
            "fps": round(self._fps_actual, 1),
            "frame_count": self._frame_count,
            "error_count": self._error_count,
            "uptime": round(self.uptime, 1),
            "queue_size": self.queue_size,
        }
