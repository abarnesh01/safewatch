from loguru import logger
from .db_manager import DatabaseManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class IncidentEvent:
    camera_id: str
    threat_type: str
    severity: str
    confidence: float
    snapshot_path: str = ""
    description: str = ""
    timestamp: datetime = datetime.now()

class IncidentLogger:
    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()

    def log_incident(self, event: IncidentEvent):
        query = """
            INSERT INTO incidents (camera_id, threat_type, severity, confidence, snapshot_path, description, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            event.camera_id,
            event.threat_type,
            event.severity,
            event.confidence,
            str(event.snapshot_path),
            event.description,
            event.timestamp.isoformat()
        )
        try:
            self.db.execute(query, params)
            logger.info(f"Incident logged: {event.threat_type} on {event.camera_id} (Severity: {event.severity})")
        except Exception as e:
            logger.error(f"Failed to log incident: {e}")

    def get_recent_incidents(self, limit: int = 50):
        query = "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?"
        return self.db.fetch_all(query, (limit,))

    def acknowledge_incident(self, incident_id: int):
        query = "UPDATE incidents SET acknowledged = 1 WHERE id = ?"
        self.db.execute(query, (incident_id,))

    def get_stats_by_type(self):
        query = "SELECT threat_type, COUNT(*) as count FROM incidents GROUP BY threat_type"
        return self.db.fetch_all(query)
