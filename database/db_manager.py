import sqlite3
import threading
from pathlib import Path
from loguru import logger
import time
from utils.runtime_isolation import RuntimePath

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
        
        # Analytics Cache
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 5.0 # 5 seconds for live analytics
        
        self._initialize_db()
        self._initialized = True
        logger.info(f"Database initialized at {self.db_path} with analytics caching")

    def fetch_cached(self, query: str, params: tuple = (), ttl: float = None):
        """Fetch from cache if available and not expired."""
        now = time.time()
        cache_key = (query, params)
        
        with self._cache_lock:
            if cache_key in self._cache:
                data, expiry = self._cache[cache_key]
                if now < expiry:
                    return data

        # Cache miss or expired
        results = self.fetch_all(query, params)
        
        with self._cache_lock:
            self._cache[cache_key] = (results, now + (ttl or self._cache_ttl))
            
        return results

    def invalidate_cache(self, query_pattern: str = None):
        """Clear cache entries matching a pattern or all if None."""
        with self._cache_lock:
            if query_pattern is None:
                self._cache.clear()
            else:
                self._cache = {k: v for k, v in self._cache.items() if query_pattern not in k[0]}

    def _get_connection(self):
        # ...
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

    def get_heatmap_data(self, camera_id: str, start_date: str, end_date: str):
        """Fetch incident coordinates for heatmap generation."""
        query = """
            SELECT threat_type, description 
            FROM incidents 
            WHERE camera_id = ? AND timestamp BETWEEN ? AND ?
        """
        rows = self.fetch_all(query, (camera_id, start_date, end_date))
        
        # We parse the description or metadata to extract coordinates
        # For now, let's assume coordinates are in the description or a separate field
        # In a real system, we'd have x, y columns
        return rows

    def get_daily_stats(self, date_str: str = None):
        pass

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
