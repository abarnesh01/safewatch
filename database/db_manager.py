"""
SafeWatch Database Manager
Thread-safe SQLite database management with connection pooling,
schema migration, and analytics support.
"""

import sqlite3
import threading
import csv
import io
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Any

from loguru import logger


@dataclass
class QueryResult:
    """Container for database query results."""
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    row_count: int = 0
    success: bool = True
    error: Optional[str] = None


class DatabaseManager:
    """Thread-safe SQLite database manager for SafeWatch."""

    _SCHEMA_VERSION = 1

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id TEXT UNIQUE NOT NULL,
        camera_id TEXT NOT NULL,
        camera_name TEXT DEFAULT '',
        threat_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        confidence REAL DEFAULT 0.0,
        risk_level TEXT DEFAULT 'LOW',
        description TEXT DEFAULT '',
        snapshot_path TEXT DEFAULT '',
        person_count INTEGER DEFAULT 0,
        zone_name TEXT DEFAULT '',
        acknowledged INTEGER DEFAULT 0,
        acknowledged_by TEXT DEFAULT '',
        acknowledged_at TEXT DEFAULT '',
        alert_sent INTEGER DEFAULT 0,
        false_positive INTEGER DEFAULT 0,
        notes TEXT DEFAULT '',
        metadata_json TEXT DEFAULT '{}',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS camera_health (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id TEXT NOT NULL,
        camera_name TEXT DEFAULT '',
        status TEXT NOT NULL DEFAULT 'offline',
        fps REAL DEFAULT 0.0,
        frame_count INTEGER DEFAULT 0,
        error_count INTEGER DEFAULT 0,
        last_error TEXT DEFAULT '',
        uptime_seconds REAL DEFAULT 0.0,
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS daily_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        camera_id TEXT DEFAULT 'all',
        total_incidents INTEGER DEFAULT 0,
        fight_count INTEGER DEFAULT 0,
        fall_count INTEGER DEFAULT 0,
        harassment_count INTEGER DEFAULT 0,
        assault_count INTEGER DEFAULT 0,
        unconscious_count INTEGER DEFAULT 0,
        trespass_count INTEGER DEFAULT 0,
        crowd_panic_count INTEGER DEFAULT 0,
        accident_count INTEGER DEFAULT 0,
        abuse_count INTEGER DEFAULT 0,
        avg_risk_score REAL DEFAULT 0.0,
        max_risk_level TEXT DEFAULT 'SAFE',
        alerts_sent INTEGER DEFAULT 0,
        false_positives INTEGER DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(date, camera_id)
    );
    CREATE TABLE IF NOT EXISTS system_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        source TEXT DEFAULT '',
        message TEXT DEFAULT '',
        severity TEXT DEFAULT 'INFO',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_incidents_camera ON incidents(camera_id);
    CREATE INDEX IF NOT EXISTS idx_incidents_threat ON incidents(threat_type);
    CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at);
    CREATE INDEX IF NOT EXISTS idx_incidents_risk ON incidents(risk_level);
    """

    def __init__(self, db_path: str, backup_interval: int = 86400) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_interval = backup_interval
        self._lock = threading.RLock()
        self._local = threading.local()
        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return self._local.conn

    def _initialize_database(self) -> None:
        with self._lock:
            try:
                conn = self._get_connection()
                conn.executescript(self._SCHEMA_SQL)
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (self._SCHEMA_VERSION,),
                )
                conn.commit()
                logger.info("Database initialized at {}", self._db_path)
            except sqlite3.Error as exc:
                logger.error("Database init failed: {}", exc)
                raise

    def execute(self, query: str, params: tuple = ()) -> QueryResult:
        result = QueryResult()
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute(query, params)
                if cursor.description:
                    result.columns = [d[0] for d in cursor.description]
                    result.rows = cursor.fetchall()
                    result.row_count = len(result.rows)
                else:
                    result.row_count = cursor.rowcount
                conn.commit()
            except sqlite3.Error as exc:
                result.success = False
                result.error = str(exc)
                logger.error("Query failed: {}", exc)
        return result

    def insert_incident(self, incident_id: str, camera_id: str, camera_name: str,
                        threat_type: str, severity: str, confidence: float,
                        risk_level: str, description: str = "",
                        snapshot_path: str = "", person_count: int = 0,
                        zone_name: str = "", metadata_json: str = "{}") -> bool:
        result = self.execute(
            """INSERT INTO incidents (incident_id, camera_id, camera_name, threat_type,
               severity, confidence, risk_level, description, snapshot_path,
               person_count, zone_name, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (incident_id, camera_id, camera_name, threat_type, severity,
             confidence, risk_level, description, snapshot_path,
             person_count, zone_name, metadata_json),
        )
        if result.success:
            logger.debug("Incident inserted: {} [{}]", incident_id, threat_type)
        return result.success

    def get_incidents(self, camera_id: str = None, threat_type: str = None,
                      severity: str = None, risk_level: str = None,
                      start_date: str = None, end_date: str = None,
                      limit: int = 100, offset: int = 0) -> list:
        conds, params = [], []
        if camera_id:
            conds.append("camera_id = ?"); params.append(camera_id)
        if threat_type:
            conds.append("threat_type = ?"); params.append(threat_type)
        if severity:
            conds.append("severity = ?"); params.append(severity)
        if risk_level:
            conds.append("risk_level = ?"); params.append(risk_level)
        if start_date:
            conds.append("created_at >= ?"); params.append(start_date)
        if end_date:
            conds.append("created_at <= ?"); params.append(end_date)
        where = " AND ".join(conds) if conds else "1=1"
        params.extend([limit, offset])
        result = self.execute(
            f"SELECT * FROM incidents WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            tuple(params),
        )
        return [dict(row) for row in result.rows] if result.success else []

    def acknowledge_incident(self, incident_id: str, by: str = "operator") -> bool:
        now = datetime.utcnow().isoformat()
        r = self.execute(
            "UPDATE incidents SET acknowledged=1, acknowledged_by=?, acknowledged_at=?, updated_at=? WHERE incident_id=?",
            (by, now, now, incident_id),
        )
        return r.success and r.row_count > 0

    def mark_false_positive(self, incident_id: str) -> bool:
        now = datetime.utcnow().isoformat()
        r = self.execute(
            "UPDATE incidents SET false_positive=1, updated_at=? WHERE incident_id=?",
            (now, incident_id),
        )
        return r.success and r.row_count > 0

    def update_camera_health(self, camera_id: str, camera_name: str, status: str,
                             fps: float = 0.0, frame_count: int = 0,
                             error_count: int = 0, last_error: str = "",
                             uptime_seconds: float = 0.0) -> bool:
        r = self.execute(
            """INSERT INTO camera_health (camera_id, camera_name, status, fps,
               frame_count, error_count, last_error, uptime_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (camera_id, camera_name, status, fps, frame_count,
             error_count, last_error, uptime_seconds),
        )
        return r.success

    def get_camera_health(self, camera_id: str = None) -> list:
        if camera_id:
            r = self.execute(
                "SELECT * FROM camera_health WHERE camera_id=? ORDER BY updated_at DESC LIMIT 1",
                (camera_id,),
            )
        else:
            r = self.execute("SELECT * FROM camera_health ORDER BY updated_at DESC")
        return [dict(row) for row in r.rows] if r.success else []

    def get_daily_stats(self, days: int = 7) -> list:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        r = self.execute(
            "SELECT * FROM daily_stats WHERE date >= ? ORDER BY date DESC", (cutoff,),
        )
        return [dict(row) for row in r.rows] if r.success else []

    def get_incident_count(self, camera_id: str = None, threat_type: str = None,
                           hours: int = 24) -> int:
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        conds, params = ["created_at >= ?"], [cutoff]
        if camera_id:
            conds.append("camera_id = ?"); params.append(camera_id)
        if threat_type:
            conds.append("threat_type = ?"); params.append(threat_type)
        where = " AND ".join(conds)
        r = self.execute(f"SELECT COUNT(*) as cnt FROM incidents WHERE {where}", tuple(params))
        if r.success and r.rows:
            return dict(r.rows[0]).get("cnt", 0) or 0
        return 0

    def export_incidents_csv(self, start_date: str = None, end_date: str = None) -> str:
        conds, params = [], []
        if start_date:
            conds.append("created_at >= ?"); params.append(start_date)
        if end_date:
            conds.append("created_at <= ?"); params.append(end_date)
        where = " AND ".join(conds) if conds else "1=1"
        r = self.execute(f"SELECT * FROM incidents WHERE {where} ORDER BY created_at DESC", tuple(params))
        output = io.StringIO()
        writer = csv.writer(output)
        if r.success and r.columns:
            writer.writerow(r.columns)
            for row in r.rows:
                writer.writerow(tuple(row))
        return output.getvalue()

    def log_system_event(self, event_type: str, source: str, message: str,
                         severity: str = "INFO") -> bool:
        r = self.execute(
            "INSERT INTO system_events (event_type, source, message, severity) VALUES (?, ?, ?, ?)",
            (event_type, source, message, severity),
        )
        return r.success

    def cleanup_old_records(self, days: int = 90) -> int:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        r = self.execute("DELETE FROM incidents WHERE created_at < ?", (cutoff,))
        deleted = r.row_count if r.success else 0
        if deleted > 0:
            logger.info("Cleaned up {} old records", deleted)
        return deleted

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
                self._local.conn = None
            except sqlite3.Error as exc:
                logger.error("Error closing db: {}", exc)
