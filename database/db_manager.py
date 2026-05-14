"""
SafeWatch — DatabaseManager
SQLite database management for incident logging and system state.
"""

import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Optional

from loguru import logger


class DatabaseManager:
    """Manages SQLite database for SafeWatch incident logging and system state."""

    def __init__(self, db_path: str = "logs/safewatch.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
        logger.info(f"DatabaseManager initialized at {self._db_path}")

    def __repr__(self) -> str:
        return f"DatabaseManager(db_path='{self._db_path}')"

    @contextmanager
    def _get_connection(self):
        """Thread-safe context manager for database connections."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Create all required tables if they don't exist."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS incidents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        threat_type TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        severity TEXT NOT NULL,
                        persons_involved INTEGER DEFAULT 0,
                        description TEXT,
                        snapshot_path TEXT,
                        recording_path TEXT,
                        alert_sent INTEGER DEFAULT 0,
                        acknowledged INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        camera_id TEXT
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS camera_status (
                        camera_id TEXT PRIMARY KEY,
                        last_seen DATETIME,
                        status TEXT DEFAULT 'offline',
                        fps REAL DEFAULT 0.0,
                        frames_processed INTEGER DEFAULT 0,
                        threats_today INTEGER DEFAULT 0
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incidents_camera
                    ON incidents(camera_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incidents_timestamp
                    ON incidents(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_incidents_threat_type
                    ON incidents(threat_type)
                """)

                logger.debug("Database tables initialized successfully")

    def log_incident(self, incident_data: dict) -> int:
        """
        Log an incident to the database.

        Args:
            incident_data: Dict with keys: camera_id, timestamp, threat_type,
                          confidence, severity, persons_involved, description,
                          snapshot_path, recording_path, alert_sent

        Returns:
            The ID of the inserted incident row.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO incidents 
                    (camera_id, timestamp, threat_type, confidence, severity,
                     persons_involved, description, snapshot_path, recording_path,
                     alert_sent, acknowledged)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    incident_data.get("camera_id", "UNKNOWN"),
                    incident_data.get("timestamp", datetime.now().isoformat()),
                    incident_data.get("threat_type", "UNKNOWN"),
                    incident_data.get("confidence", 0.0),
                    incident_data.get("severity", "LOW"),
                    incident_data.get("persons_involved", 0),
                    incident_data.get("description", ""),
                    incident_data.get("snapshot_path", ""),
                    incident_data.get("recording_path", ""),
                    incident_data.get("alert_sent", 0),
                ))
                incident_id = cursor.lastrowid
                logger.info(
                    f"Incident logged: id={incident_id} type={incident_data.get('threat_type')} "
                    f"camera={incident_data.get('camera_id')} severity={incident_data.get('severity')}"
                )
                return incident_id

    def get_incidents(
        self,
        camera_id: Optional[str] = None,
        threat_type: Optional[str] = None,
        severity: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Get incidents with optional filters.

        Args:
            camera_id: Filter by camera ID
            threat_type: Filter by threat type
            severity: Filter by severity level
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Max number of results
            offset: Offset for pagination

        Returns:
            List of incident dicts
        """
        query = "SELECT * FROM incidents WHERE 1=1"
        params: list = []

        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if threat_type:
            query += " AND threat_type = ?"
            params.append(threat_type)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_daily_stats(self, date: Optional[str] = None) -> dict:
        """
        Get threat counts by type and camera for a given date.

        Args:
            date: Date string in YYYY-MM-DD format. Defaults to today.

        Returns:
            Dict with keys: total_incidents, by_type, by_camera, by_severity
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        start = f"{date} 00:00:00"
        end = f"{date} 23:59:59"

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT COUNT(*) as total FROM incidents WHERE timestamp BETWEEN ? AND ?",
                    (start, end),
                )
                total = cursor.fetchone()["total"]

                cursor.execute(
                    "SELECT threat_type, COUNT(*) as count FROM incidents "
                    "WHERE timestamp BETWEEN ? AND ? GROUP BY threat_type",
                    (start, end),
                )
                by_type = {row["threat_type"]: row["count"] for row in cursor.fetchall()}

                cursor.execute(
                    "SELECT camera_id, COUNT(*) as count FROM incidents "
                    "WHERE timestamp BETWEEN ? AND ? GROUP BY camera_id",
                    (start, end),
                )
                by_camera = {row["camera_id"]: row["count"] for row in cursor.fetchall()}

                cursor.execute(
                    "SELECT severity, COUNT(*) as count FROM incidents "
                    "WHERE timestamp BETWEEN ? AND ? GROUP BY severity",
                    (start, end),
                )
                by_severity = {row["severity"]: row["count"] for row in cursor.fetchall()}

                return {
                    "date": date,
                    "total_incidents": total,
                    "by_type": by_type,
                    "by_camera": by_camera,
                    "by_severity": by_severity,
                }

    def get_recent_incidents(self, n: int = 20) -> list[dict]:
        """Get the last N incidents ordered by timestamp descending."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?", (n,)
                )
                return [dict(row) for row in cursor.fetchall()]

    def update_camera_status(self, camera_id: str, status_data: dict):
        """
        Insert or update camera status.

        Args:
            camera_id: Camera identifier
            status_data: Dict with keys: status, fps, frames_processed, threats_today
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO camera_status (camera_id, last_seen, status, fps, frames_processed, threats_today)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET
                        last_seen = excluded.last_seen,
                        status = excluded.status,
                        fps = excluded.fps,
                        frames_processed = excluded.frames_processed,
                        threats_today = excluded.threats_today
                """, (
                    camera_id,
                    datetime.now().isoformat(),
                    status_data.get("status", "unknown"),
                    status_data.get("fps", 0.0),
                    status_data.get("frames_processed", 0),
                    status_data.get("threats_today", 0),
                ))

    def get_camera_status(self, camera_id: Optional[str] = None) -> list[dict]:
        """Get status of one or all cameras."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if camera_id:
                    cursor.execute(
                        "SELECT * FROM camera_status WHERE camera_id = ?", (camera_id,)
                    )
                else:
                    cursor.execute("SELECT * FROM camera_status")
                return [dict(row) for row in cursor.fetchall()]

    def log_system_event(self, level: str, message: str, camera_id: Optional[str] = None):
        """Log a system event."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO system_logs (timestamp, level, message, camera_id) VALUES (?, ?, ?, ?)",
                    (datetime.now().isoformat(), level, message, camera_id),
                )

    def get_system_logs(self, limit: int = 100, level: Optional[str] = None) -> list[dict]:
        """Get system logs with optional level filter."""
        query = "SELECT * FROM system_logs WHERE 1=1"
        params: list = []
        if level:
            query += " AND level = ?"
            params.append(level)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]

    def acknowledge_incident(self, incident_id: int) -> bool:
        """Mark an incident as acknowledged."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE incidents SET acknowledged = 1 WHERE id = ?", (incident_id,)
                )
                updated = cursor.rowcount > 0
                if updated:
                    logger.info(f"Incident {incident_id} acknowledged")
                return updated

    def get_hourly_distribution(self, date: Optional[str] = None) -> dict:
        """Get incident counts per hour for charts."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        start = f"{date} 00:00:00"
        end = f"{date} 23:59:59"

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
                    FROM incidents
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY hour
                    ORDER BY hour
                """, (start, end))
                result = {str(h).zfill(2): 0 for h in range(24)}
                for row in cursor.fetchall():
                    result[row["hour"]] = row["count"]
                return result

    def backup(self, backup_dir: str = "logs/backups"):
        """Create a backup of the database."""
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = backup_path / f"safewatch_backup_{timestamp}.db"

        with self._lock:
            with self._get_connection() as conn:
                backup_conn = sqlite3.connect(str(dest))
                conn.backup(backup_conn)
                backup_conn.close()
                logger.info(f"Database backed up to {dest}")

    def cleanup_old_incidents(self, retention_days: int = 30):
        """Delete incidents older than retention_days."""
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM incidents WHERE timestamp < ?", (cutoff,)
                )
                deleted = cursor.rowcount
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old incidents (older than {retention_days} days)")
