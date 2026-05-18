"""
SafeWatch — CameraStream
Threaded camera capture with buffering, reconnection, and FPS tracking.
"""

import time
import threading
from queue import Queue, Full, Empty
from typing import Optional, Union

import cv2
import numpy as np
from loguru import logger
from capture.video_buffer import RollingVideoBuffer


class CameraStream:
    """
    Threaded camera stream with automatic reconnection and frame buffering.
    Supports USB webcam indices and RTSP URLs.
    """

    def __init__(
        self,
        camera_id: str,
        source: Union[int, str],
        resolution: tuple[int, int] = (640, 480),
        fps_target: int = 15,
        buffer_size: int = 128,
        reconnect_delay: float = 5.0,
        name: str = "Camera",
    ):
        self._camera_id = camera_id
        self._source = source
        self._resolution = resolution
        self._fps_target = fps_target
        self._name = name
        self._buffer: Queue = Queue(maxsize=buffer_size)
        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._lock = threading.Lock()
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._last_frame_time = 0.0
        self._reconnect_delay = reconnect_delay
        self._last_read_frame: Optional[np.ndarray] = None
        self._reconnect_count = 0
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_interval = 10.0
        
        # New Video Buffer System
        self.video_buffer = RollingVideoBuffer(fps=fps_target)
        
        logger.info(f"CameraStream created: id={camera_id} source={source} name={name}")

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        conn = "connected" if self._connected else "disconnected"
        return (
            f"CameraStream(id='{self._camera_id}', source={self._source}, "
            f"status={status}, conn={conn}, fps={self._fps:.1f})"
        )

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def name(self) -> str:
        return self._name

    def _open_capture(self) -> bool:
        """Open the video capture device."""
        try:
            if isinstance(self._source, int):
                self._capture = cv2.VideoCapture(self._source)
            else:
                self._capture = cv2.VideoCapture(self._source, cv2.CAP_FFMPEG)

            if self._capture is None or not self._capture.isOpened():
                logger.warning(f"[{self._camera_id}] Failed to open source: {self._source}")
                self._connected = False
                return False

            # Use MJPG codec for USB webcams — prevents V4L2 YUYV timeout
            if isinstance(self._source, int):
                self._capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'),
                )

            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])

            if isinstance(self._source, int):
                self._capture.set(cv2.CAP_PROP_FPS, self._fps_target)

            # Warm-up: let the camera sensor stabilize
            import time as _time
            _time.sleep(0.5)

            self._connected = True
            logger.info(f"[{self._camera_id}] Camera opened successfully: {self._source}")
            return True

        except Exception as e:
            logger.error(f"[{self._camera_id}] Error opening camera: {e}")
            self._connected = False
            return False

    def _capture_loop(self):
        """Main capture loop with jitter stabilization and stale frame dropping."""
        logger.info(f"[{self._camera_id}] Capture thread started")
        frame_interval = 1.0 / self._fps_target if self._fps_target > 0 else 0.033
        
        # Jitter stabilization parameters
        is_rtsp = isinstance(self._source, str) and "rtsp" in self._source.lower()
        max_latency = 2.0  # seconds
        
        while self._running:
            if not self._connected or self._capture is None or not self._capture.isOpened():
                logger.warning(f"[{self._camera_id}] Connection lost, attempting reconnect in {self._reconnect_delay}s...")
                self._connected = False
                if self._capture is not None:
                    try:
                        self._capture.release()
                    except Exception:
                        pass
                time.sleep(self._reconnect_delay)
                if self._running:
                    self._open_capture()
                continue

            try:
                ret, frame = self._capture.read()
                now = time.time()
                
                if not ret or frame is None:
                    logger.warning(f"[{self._camera_id}] Failed to read frame")
                    self._connected = False
                    continue

                # 1. Stale frame dropping for RTSP (prevent latency accumulation)
                if is_rtsp:
                    # If we are falling behind significantly, skip frames
                    if now - self._last_frame_time > max_latency:
                        logger.debug(f"[{self._camera_id}] Dropping stale RTSP frames (latency={now - self._last_frame_time:.2f}s)")
                        # Read and discard until we catch up or buffer is empty
                        for _ in range(5):
                            self._capture.grab()
                        continue

                # 2. Frame processing
                if frame.shape[1] != self._resolution[0] or frame.shape[0] != self._resolution[1]:
                    frame = cv2.resize(frame, self._resolution)

                # Append to rolling video evidence buffer
                self.video_buffer.append_frame(frame)

                self._last_frame_time = now

                # 3. Buffer management with jitter protection
                try:
                    # For unstable WiFi, we use a simple "most-recent-only" strategy in the main buffer
                    # but ensure we don't burst too fast
                    self._buffer.put_nowait(frame)
                except Full:
                    try:
                        self._buffer.get_nowait()
                    except Empty:
                        pass
                    try:
                        self._buffer.put_nowait(frame)
                    except Full:
                        pass

                self._frame_count += 1
                elapsed = now - self._fps_timer
                if elapsed >= 1.0:
                    with self._lock:
                        self._fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_timer = now

                # 4. Latency stabilization sleep
                # If we read a frame extremely fast, we sleep to match target FPS
                process_time = time.time() - now
                sleep_time = frame_interval - process_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"[{self._camera_id}] Capture error: {e}")
                self._connected = False
                time.sleep(1.0)

        logger.info(f"[{self._camera_id}] Capture thread stopped")

    def start(self):
        """Start the camera capture thread."""
        if self._running:
            logger.warning(f"[{self._camera_id}] Already running")
            return

        self._running = True
        if not self._open_capture():
            logger.warning(f"[{self._camera_id}] Initial connection failed, will retry in background")

        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"CameraStream-{self._camera_id}",
            daemon=True,
        )
        self._thread.start()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name=f"Watchdog-{self._camera_id}",
            daemon=True,
        )
        self._watchdog_thread.start()
        logger.info(f"[{self._camera_id}] Camera stream and watchdog started")

    def _watchdog_loop(self):
        """Watchdog to detect and recover from frozen streams."""
        while self._running:
            time.sleep(self._watchdog_interval)
            if not self._running: break

            now = time.time()
            if self._connected and (now - self._last_frame_time > self._watchdog_interval):
                logger.warning(f"[{self._camera_id}] Watchdog detected stalled stream (no frames for {now - self._last_frame_time:.1f}s). Restarting...")
                self._reconnect_count += 1
                self._connected = False
                if self._capture:
                    try:
                        self._capture.release()
                    except:
                        pass

    def stop(self):
        """Stop the camera capture thread and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

        self._connected = False

        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except Empty:
                break

        logger.info(f"[{self._camera_id}] Camera stream stopped")

    def read(self) -> Optional[np.ndarray]:
        """
        Read the latest frame from the buffer.

        Returns:
            The latest frame as a numpy array, or None if no frame available.
        """
        frame = None
        try:
            while not self._buffer.empty():
                frame = self._buffer.get_nowait()
        except Empty:
            pass
        if frame is not None:
            self._last_read_frame = frame
        return self._last_read_frame

    def get_fps(self) -> float:
        """Get the current frames per second rate."""
        with self._lock:
            return round(self._fps, 1)

    def is_running(self) -> bool:
        """Check if the capture thread is running."""
        return self._running

    def is_connected(self) -> bool:
        """Check if the camera is connected and producing frames."""
        return self._connected

    def get_status(self) -> dict:
        """Get comprehensive camera status."""
        return {
            "camera_id": self._camera_id,
            "name": self._name,
            "source": self._source,
            "running": self._running,
            "connected": self._connected,
            "fps": self.get_fps(),
            "buffer_size": self._buffer.qsize(),
            "resolution": self._resolution,
            "last_frame_time": self._last_frame_time,
            "reconnect_count": self._reconnect_count,
        }
