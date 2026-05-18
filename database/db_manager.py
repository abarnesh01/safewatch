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
        
        # Write Queue for Concurrency Hardening
        from queue import Queue
        self._write_queue = Queue()
        self._write_thread = threading.Thread(target=self._write_worker, daemon=True)
        self._write_thread.start()
        
        self._initialize_db()
        self._initialized = True
        logger.info(f"Database hardened: WAL enabled, WriteQueue active at {self.db_path}")

    def _write_worker(self):
        """Background worker to serialize all database writes."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        while True:
            try:
                query, params = self._write_queue.get()
                if query is None: break
                
                # Retry logic for locked database
                for attempt in range(5):
                    try:
                        conn.execute(query, params)
                        conn.commit()
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e).lower():
                            time.sleep(0.05 * (2 ** attempt))
                        else:
                            raise
                self._write_queue.task_done()
            except Exception as e:
                logger.error(f"Database write worker error: {e}")

    def execute_async(self, query: str, params: tuple = ()):
        """Queue a write operation to be executed in the background."""
        self._write_queue.put((query, params))

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
        
        # Enable WAL mode for high-concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        
        # Incidents table (v2 - Forensic Intelligence)
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
                acknowledged INTEGER DEFAULT 0,
                correlation_id TEXT,
                parent_incident_id INTEGER,
                operator_notes TEXT,
                tags TEXT,
                metadata_json TEXT,
                video_evidence_path TEXT
            )
        ''')
        
        # Optimization Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inc_ts ON incidents(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inc_cam ON incidents(camera_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inc_sev ON incidents(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inc_corr ON incidents(correlation_id)")

        # Migration: Ensure new columns exist for v1 databases
        try:
            cursor.execute("ALTER TABLE incidents ADD COLUMN correlation_id TEXT")
            cursor.execute("ALTER TABLE incidents ADD COLUMN parent_incident_id INTEGER")
            cursor.execute("ALTER TABLE incidents ADD COLUMN operator_notes TEXT")
            cursor.execute("ALTER TABLE incidents ADD COLUMN tags TEXT")
            cursor.execute("ALTER TABLE incidents ADD COLUMN metadata_json TEXT")
        except sqlite3.OperationalError:
            pass # Columns already exist
            
        try:
            cursor.execute("ALTER TABLE incidents ADD COLUMN video_evidence_path TEXT")
        except sqlite3.OperationalError:
            pass # Column already exists
            
        # Phase 3 Federation Migrations
        try:
            cursor.execute("ALTER TABLE incidents ADD COLUMN sync_status TEXT DEFAULT 'PENDING'")
        except sqlite3.OperationalError:
            pass

        # Audit Logs (Enterprise Requirement)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                operator_id TEXT,
                action TEXT,
                target_id TEXT,
                details TEXT,
                ip_address TEXT
            )
        ''')
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_logs(timestamp)")
        
        # System health/logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                message TEXT
            )
        ''')
        
        # Phase 2: Face Recognition & Analytics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                category TEXT NOT NULL,
                embedding BLOB,
                image_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                person_name TEXT,
                category TEXT,
                confidence REAL,
                snapshot_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS heatmap_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                x INTEGER,
                y INTEGER,
                risk_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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

    def prune_old_data(self, days: int = 30):
        """Prune incident and audit records older than N days."""
        logger.info(f"Database Pruning: Removing records older than {days} days...")
        
        queries = [
            "DELETE FROM incidents WHERE timestamp < date('now', '-' || ? || ' days')",
            "DELETE FROM audit_logs WHERE timestamp < date('now', '-' || ? || ' days')",
            "DELETE FROM system_logs WHERE timestamp < date('now', '-' || ? || ' days')"
        ]
        
        try:
            for q in queries:
                self.execute_async(q, (str(days),))
            logger.info("Database pruning tasks queued.")
        except Exception as e:
            logger.error(f"Database pruning failed: {e}")

    def close(self):
        # Stop write worker
        self._write_queue.put((None, None))
        if self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
            
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
