"""
SafeWatch — IncidentLogger
High-level wrapper around DatabaseManager for threat incident logging.
"""

import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from database.db_manager import DatabaseManager


class IncidentLogger:
    """High-level incident logging and querying interface wrapping DatabaseManager."""

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager
        self._incident_counter = 0
        logger.info("IncidentLogger initialized")

    def __repr__(self) -> str:
        return f"IncidentLogger(db={self._db!r})"

    def log_threat(
        self,
        threat_event: dict,
        camera_id: str,
        snapshot_path: str = "",
        recording_path: str = "",
    ) -> int:
        """
        Log a threat event to the database.

        Args:
            threat_event: Dict with keys: threat_type, confidence, severity,
                         persons_involved, description
            camera_id: Camera that detected the threat
            snapshot_path: Path to saved snapshot image
            recording_path: Path to saved video recording

        Returns:
            The incident ID
        """
        self._incident_counter += 1
        incident_data = {
            "camera_id": camera_id,
            "timestamp": threat_event.get("timestamp", datetime.now().isoformat()),
            "threat_type": threat_event.get("threat_type", "UNKNOWN"),
            "confidence": threat_event.get("confidence", 0.0),
            "severity": threat_event.get("severity", "LOW"),
            "persons_involved": len(threat_event.get("persons_involved", [])),
            "description": threat_event.get("description", ""),
            "snapshot_path": snapshot_path,
            "recording_path": recording_path,
            "alert_sent": threat_event.get("alert_sent", 0),
        }
        incident_id = self._db.log_incident(incident_data)
        logger.info(
            f"Threat logged: incident_id={incident_id} type={incident_data['threat_type']} "
            f"camera={camera_id} severity={incident_data['severity']}"
        )
        return incident_id

    def get_threat_stats(self, last_hours: int = 24) -> dict:
        """
        Get threat statistics for the last N hours.

        Args:
            last_hours: Number of hours to look back

        Returns:
            Dict with keys: total, by_type, by_severity, by_camera, avg_confidence
        """
        start_time = (datetime.now() - timedelta(hours=last_hours)).isoformat()
        incidents = self._db.get_incidents(start_date=start_time, limit=10000)

        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_camera: dict[str, int] = {}
        total_confidence = 0.0

        for inc in incidents:
            t = inc.get("threat_type", "UNKNOWN")
            s = inc.get("severity", "LOW")
            c = inc.get("camera_id", "UNKNOWN")

            by_type[t] = by_type.get(t, 0) + 1
            by_severity[s] = by_severity.get(s, 0) + 1
            by_camera[c] = by_camera.get(c, 0) + 1
            total_confidence += inc.get("confidence", 0.0)

        total = len(incidents)
        avg_confidence = total_confidence / total if total > 0 else 0.0

        return {
            "total": total,
            "last_hours": last_hours,
            "by_type": by_type,
            "by_severity": by_severity,
            "by_camera": by_camera,
            "avg_confidence": round(avg_confidence, 3),
        }

    def get_timeline(self, camera_id: str, date: Optional[str] = None) -> list[dict]:
        """
        Get an ordered timeline of incidents for a camera on a specific date.

        Args:
            camera_id: Camera ID to filter by
            date: Date in YYYY-MM-DD format. Defaults to today.

        Returns:
            Ordered list of incidents for the camera on the given date.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        start_date = f"{date} 00:00:00"
        end_date = f"{date} 23:59:59"

        incidents = self._db.get_incidents(
            camera_id=camera_id,
            start_date=start_date,
            end_date=end_date,
            limit=1000,
        )
        return sorted(incidents, key=lambda x: x.get("timestamp", ""))

    def export_csv(self, start_date: str, end_date: str, output_path: str) -> str:
        """
        Export incidents to a CSV file.

        Args:
            start_date: Start date (ISO format or YYYY-MM-DD)
            end_date: End date (ISO format or YYYY-MM-DD)
            output_path: Path for the output CSV file

        Returns:
            The absolute path of the created CSV file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        incidents = self._db.get_incidents(
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )

        fieldnames = [
            "id", "camera_id", "timestamp", "threat_type", "confidence",
            "severity", "persons_involved", "description", "snapshot_path",
            "recording_path", "alert_sent", "acknowledged", "created_at",
        ]

        with open(out, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for inc in incidents:
                writer.writerow(inc)

        logger.info(f"Exported {len(incidents)} incidents to {out}")
        return str(out.resolve())

    def mark_alert_sent(self, incident_id: int):
        """Mark that an alert was sent for an incident."""
        self._db.log_system_event(
            level="INFO",
            message=f"Alert sent for incident {incident_id}",
        )

    def get_unacknowledged(self, limit: int = 50) -> list[dict]:
        """Get recent unacknowledged incidents."""
        all_recent = self._db.get_recent_incidents(n=limit * 2)
        return [inc for inc in all_recent if inc.get("acknowledged", 0) == 0][:limit]
