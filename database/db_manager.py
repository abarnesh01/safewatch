import sqlite3
import threading
from pathlib import Path
from loguru import logger
from datetime import datetime

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = "logs/safewatch.db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = "logs/safewatch.db"):
        if self._initialized:
            return
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._initialize_db()
        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")

    def _get_connection(self):
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    def _initialize_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Incidents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera_id TEXT,
                threat_type TEXT,
                severity TEXT,
                confidence REAL,
                snapshot_path TEXT,
                description TEXT,
                acknowledged INTEGER DEFAULT 0
            )
        ''')
        
        # System health/logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                message TEXT
            )
        ''')
        
        conn.commit()

    def execute(self, query: str, params: tuple = ()):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor
        except sqlite3.Error as e:
            logger.error(f"Database error: {e} | Query: {query}")
            raise

    def fetch_all(self, query: str, params: tuple = ()):
        cursor = self.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
