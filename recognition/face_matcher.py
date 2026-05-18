import numpy as np
from loguru import logger

class FaceMatcher:
    """Performs cosine similarity matching against the active watchlist."""
    
    def __init__(self, watchlist_manager):
        self.wl_manager = watchlist_manager
        
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculates cosine similarity between two feature vectors."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def match_face(self, embedding: np.ndarray) -> dict:
        """
        Compares an unknown embedding against the watchlist.
        Returns categorization (MATCH, REVIEW, UNKNOWN).
        """
        watchlist = self.wl_manager.get_watchlist()
        
        default_resp = {
            "person_name": "Unknown", 
            "category": "UNKNOWN", 
            "confidence": 0.0, 
            "face_id": None,
            "status": "UNKNOWN"
        }
        
        if not watchlist:
            return default_resp
            
        best_match = None
        best_score = -1.0
        
        for record in watchlist:
            score = self._cosine_similarity(embedding, record["embedding"])
            if score > best_score:
                best_score = score
                best_match = record
                
        if best_match:
            # Classification logic requested by Enterprise spec
            if best_score > 0.85:
                status = "MATCH"
            elif best_score > 0.65:
                status = "REVIEW"
            else:
                status = "UNKNOWN"
                
            return {
                "person_name": best_match["name"] if status != "UNKNOWN" else "Unknown",
                "category": best_match["category"] if status != "UNKNOWN" else "UNKNOWN",
                "confidence": float(best_score),
                "face_id": best_match["id"],
                "status": status
            }
            
        return default_resp
