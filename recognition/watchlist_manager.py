import numpy as np
from loguru import logger

class WatchlistManager:
    """Manages the persistence of facial identities into the SQLite database."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        
    def _array_to_blob(self, array: np.ndarray) -> bytes:
        return array.astype(np.float32).tobytes()
        
    def _blob_to_array(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def enroll_face(self, name: str, category: str, embedding: np.ndarray, image_path: str = ""):
        """Saves a new identity and their 512D embedding vector to the DB."""
        blob = self._array_to_blob(embedding)
        try:
            self.db.execute(
                "INSERT INTO watchlists (person_name, category, embedding, image_path) VALUES (?, ?, ?, ?)",
                (name, category, blob, image_path)
            )
            logger.info(f"Successfully enrolled {name} into watchlist ({category}).")
            return True
        except Exception as e:
            logger.error(f"Failed to enroll face: {e}")
            return False

    def get_watchlist(self) -> list:
        """Retrieves all active watchlist embeddings."""
        cursor = self.db.execute("SELECT id, person_name, category, embedding FROM watchlists")
        if not cursor:
            return []
            
        watchlist = []
        for row in cursor.fetchall():
            try:
                watchlist.append({
                    "id": row[0],
                    "name": row[1],
                    "category": row[2],
                    "embedding": self._blob_to_array(row[3])
                })
            except Exception:
                continue
        return watchlist
