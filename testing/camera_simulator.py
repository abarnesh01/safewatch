import time
import threading
import numpy as np
import cv2
from loguru import logger

class CameraSimulator:
    """Simulates hundreds of concurrent RTSP feeds to stress test the ingestion pipeline."""
    
    def __init__(self, num_cameras=50, target_fps=15):
        self.num_cameras = num_cameras
        self.target_fps = target_fps
        self.running = False
        self.dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
    def start(self):
        self.running = True
        logger.info(f"Starting load generation for {self.num_cameras} virtual cameras at {self.target_fps} FPS.")
        for i in range(self.num_cameras):
            threading.Thread(target=self._simulate_feed, args=(f"sim_cam_{i}",), daemon=True).start()
            
    def _simulate_feed(self, cam_id):
        sleep_time = 1.0 / self.target_fps
        while self.running:
            # Simulate network latency/jitter
            jitter = np.random.uniform(-0.01, 0.02)
            time.sleep(max(0, sleep_time + jitter))
            
            # Send frame to StreamManager (pseudo-code hook)
            # stream_manager.push_frame(cam_id, self.dummy_frame)
            
    def stop(self):
        self.running = False
