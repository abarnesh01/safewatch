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
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str = ""
    parent_incident_id: int = 0
    tags: str = ""
    metadata: dict = field(default_factory=dict)

class IncidentLogger:
    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager or DatabaseManager()

    def log_incident(self, event: IncidentEvent):
        import json
        query = """
            INSERT INTO incidents (
                camera_id, threat_type, severity, confidence, 
                snapshot_path, description, timestamp,
                correlation_id, parent_incident_id, tags, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            event.camera_id,
            event.threat_type,
            event.severity,
            event.confidence,
            str(event.snapshot_path),
            event.description,
            event.timestamp.isoformat(),
            event.correlation_id,
            event.parent_incident_id,
            event.tags,
            json.dumps(event.metadata)
        )
        try:
            self.db.execute(query, params)
            logger.info(f"Incident logged: {event.threat_type} on {event.camera_id} (CorrID: {event.correlation_id})")
        except Exception as e:
            logger.error(f"Failed to log incident: {e}")

    def add_audit_log(self, operator_id: str, action: str, target_id: str = "", details: str = ""):
        query = "INSERT INTO audit_logs (operator_id, action, target_id, details) VALUES (?, ?, ?, ?)"
        self.db.execute(query, (operator_id, action, target_id, details))
        logger.debug(f"Audit Log: {operator_id} performed {action} on {target_id}")

    def get_recent_incidents(self, limit: int = 50):
        query = "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?"
        return self.db.fetch_all(query, (limit,))

    def acknowledge_incident(self, incident_id: int):
        query = "UPDATE incidents SET acknowledged = 1 WHERE id = ?"
        self.db.execute(query, (incident_id,))

    def get_stats_by_type(self):
        query = "SELECT threat_type, COUNT(*) as count FROM incidents GROUP BY threat_type"
        return self.db.fetch_all(query)
    def export_forensic_bundle(self, incident_id: int, output_dir: str = "exports"):
        """Export an incident package with snapshot, metadata, and forensic manifest."""
        import json
        import zipfile
        
        query = "SELECT * FROM incidents WHERE id = ?"
        incident = self.db.fetch_all(query, (incident_id,))
        if not incident:
            logger.error(f"Incident {incident_id} not found for export")
            return None
        
        incident = incident[0]
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        bundle_name = f"forensic_inc_{incident_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        bundle_path = out_path / bundle_name
        
        try:
            with zipfile.ZipFile(bundle_path, 'w') as zipf:
                # 1. Metadata JSON
                meta_json = json.dumps(dict(incident), indent=4)
                zipf.writestr("metadata.json", meta_json)
                
                # 2. Snapshot
                snap_path = Path(incident.get("snapshot_path", ""))
                if snap_path.exists():
                    zipf.write(snap_path, arcname=snap_path.name)
                
                # 3. Forensic Manifest
                manifest = f"""
                SAFEWATCH FORENSIC MANIFEST
                ===========================
                Incident ID: {incident_id}
                Export Date: {datetime.now().isoformat()}
                Camera: {incident.get('camera_id')}
                Type: {incident.get('threat_type')}
                Severity: {incident.get('severity')}
                Chain of Custody: SafeWatch Auto-Export V1
                """
                zipf.writestr("manifest.txt", manifest)
            
            logger.info(f"Forensic bundle exported: {bundle_path}")
            return str(bundle_path)
        except Exception as e:
            logger.error(f"Forensic export failed: {e}")
            return None
