import sqlite3
import numpy as np
from loguru import logger

class HeatmapGenerator:
    """Records person coordinates and generates spatial heatmap points."""
    
    def __init__(self, db_manager):
        self.db = db_manager

    def record_coordinates(self, camera_id, persons, risk_score=0.0):
        """Persists center coordinates of detected persons for footprint analytics."""
        if not persons:
            return
            
        for p in persons:
            if hasattr(p, 'box') and p.box is not None:
                x1, y1, x2, y2 = p.box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                try:
                    self.db.execute(
                        "INSERT INTO heatmap_data (camera_id, x, y, risk_score) VALUES (?, ?, ?, ?)",
                        (camera_id, cx, cy, risk_score)
                    )
                except Exception as e:
                    logger.debug(f"Heatmap save failed: {e}")

    def generate_heatmap(self, camera_id: str, time_range_hours: int = 24):
        """Fetches accumulated footprint coordinates over a time window."""
        query = f"SELECT x, y, risk_score FROM heatmap_data WHERE camera_id = ? AND timestamp >= datetime('now', '-{time_range_hours} hours')"
        
        try:
            cursor = self.db.execute(query, (camera_id,))
            points = []
            if cursor:
                for row in cursor.fetchall():
                    points.append({"x": row[0], "y": row[1], "weight": max(1.0, row[2] * 10)})
            return points
        except Exception as e:
            logger.error(f"Failed to fetch heatmap data: {e}")
            return []
