import cv2
import numpy as np
from loguru import logger
import insightface
from insightface.app import FaceAnalysis

class FaceRecognitionSystem:
    """Enterprise face detection, alignment, and embedding extraction using InsightFace."""
    
    def __init__(self, use_gpu: bool = False):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        # Initialize InsightFace model pack 'buffalo_l' (includes det and rec models)
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
        logger.info(f"FaceRecognitionSystem (InsightFace) initialized. GPU: {use_gpu}")

    def detect_and_embed(self, frame: np.ndarray):
        """
        Processes a raw frame to extract faces.
        Returns a list of Face objects which contain:
        - bbox: bounding box
        - kps: facial landmarks
        - det_score: detection confidence
        - embedding: 512D float array
        """
        try:
            faces = self.app.get(frame)
            return faces
        except Exception as e:
            logger.error(f"InsightFace processing failed: {e}")
            return []
