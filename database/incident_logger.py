"""
SafeWatch Incident Logger
High-level incident logging interface with deduplication and cooldown.
"""

import json
import uuid
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

from loguru import logger

from database.db_manager import DatabaseManager


class IncidentLogger:
    """Logs threat incidents with deduplication, cooldown, and analytics."""

    def __init__(self, db_manager: DatabaseManager, default_cooldown: int = 30) -> None:
        self._db = db_manager
        self._default_cooldown = default_cooldown
        self._cooldown_map: dict = {}
        self._lock = Lock()
        logger.info("IncidentLogger initialized (cooldown={}s)", default_cooldown)

    def _generate_incident_id(self) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return f"SW-{ts}-{short_uuid}"

    def _is_on_cooldown(self, camera_id: str, threat_type: str, cooldown: int) -> bool:
        key = f"{camera_id}:{threat_type}"
        with self._lock:
            last_time = self._cooldown_map.get(key)
            if last_time and (datetime.utcnow() - last_time).total_seconds() < cooldown:
                return True
            return False

    def _update_cooldown(self, camera_id: str, threat_type: str) -> None:
        key = f"{camera_id}:{threat_type}"
        with self._lock:
            self._cooldown_map[key] = datetime.utcnow()

    def log_incident(self, camera_id: str, camera_name: str, threat_type: str,
                     severity: str, confidence: float, risk_level: str,
                     description: str = "", snapshot_path: str = "",
                     person_count: int = 0, zone_name: str = "",
                     cooldown: Optional[int] = None,
                     metadata: Optional[dict] = None) -> Optional[str]:
        cd = cooldown if cooldown is not None else self._default_cooldown
        if self._is_on_cooldown(camera_id, threat_type, cd):
            logger.debug("Incident suppressed (cooldown): {} on {}", threat_type, camera_id)
            return None

        incident_id = self._generate_incident_id()
        metadata_json = json.dumps(metadata) if metadata else "{}"

        success = self._db.insert_incident(
            incident_id=incident_id,
            camera_id=camera_id,
            camera_name=camera_name,
            threat_type=threat_type,
            severity=severity,
            confidence=confidence,
            risk_level=risk_level,
            description=description,
            snapshot_path=snapshot_path,
            person_count=person_count,
            zone_name=zone_name,
            metadata_json=metadata_json,
        )

        if success:
            self._update_cooldown(camera_id, threat_type)
            logger.info(
                "Incident logged: {} | {} | {} | conf={:.2f} | risk={}",
                incident_id, camera_id, threat_type, confidence, risk_level,
            )
            return incident_id

        logger.error("Failed to log incident for {} on {}", threat_type, camera_id)
        return None

    def get_recent_incidents(self, hours: int = 1, camera_id: str = None,
                             threat_type: str = None, limit: int = 50) -> list:
        start = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        return self._db.get_incidents(
            camera_id=camera_id,
            threat_type=threat_type,
            start_date=start,
            limit=limit,
        )

    def get_stats_summary(self, hours: int = 24) -> dict:
        total = self._db.get_incident_count(hours=hours)
        threat_types = [
            "fight", "fall", "harassment", "assault", "unconscious",
            "trespass", "crowd_panic", "accident", "abuse",
        ]
        by_type = {}
        for tt in threat_types:
            by_type[tt] = self._db.get_incident_count(threat_type=tt, hours=hours)
        return {"total": total, "by_type": by_type, "period_hours": hours}

    def mark_alert_sent(self, incident_id: str) -> bool:
        now = datetime.utcnow().isoformat()
        r = self._db.execute(
            "UPDATE incidents SET alert_sent=1, updated_at=? WHERE incident_id=?",
            (now, incident_id),
        )
        return r.success and r.row_count > 0

    def clear_cooldowns(self) -> None:
        with self._lock:
            self._cooldown_map.clear()
            logger.info("All incident cooldowns cleared")
