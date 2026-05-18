import collections
import cv2
import threading
from loguru import logger

class RollingVideoBuffer:
    """Maintains a rolling buffer of frames to export video evidence of incidents."""
    
    def __init__(self, fps: int = 15, pre_seconds: int = 7, post_seconds: int = 8):
        self.fps = fps
        self.pre_frames = fps * pre_seconds
        self.post_frames = fps * post_seconds
        self.buffer = collections.deque(maxlen=self.pre_frames)
        self.lock = threading.Lock()
        
    def append_frame(self, frame):
        """Append a raw frame to the rolling buffer."""
        with self.lock:
            self.buffer.append(frame.copy())
            
    def export_evidence(self, post_event_frames: list, output_path: str, frame_size: tuple):
        """Export the buffered pre-event frames and the post-event frames to an MP4 file."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, frame_size)
            
            with self.lock:
                for frame in self.buffer:
                    out.write(frame)
                    
            for frame in post_event_frames[:self.post_frames]:
                out.write(frame)
                
            out.release()
            logger.info(f"Video evidence exported to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export video evidence: {e}")
            return None
