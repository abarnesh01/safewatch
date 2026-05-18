import threading
import time
from pathlib import Path
from loguru import logger

class RetentionManager:
    """Enforces GDPR/CCPA data lifecycle policies by archiving or deleting aging records."""
    
    def __init__(self, db_manager, run_interval_hours: int = 24):
        self.db = db_manager
        self.run_interval_hours = run_interval_hours
        self.running = False
        
        # Policy: Table -> max age in days
        self.policies = {
            "incidents": 90,
            "face_events": 60,
            "heatmap_data": 180,
            "system_logs": 30,
            "audit_logs": 365
        }
        
    def start(self):
        self.running = True
        threading.Thread(target=self._retention_worker, daemon=True).start()
        logger.info("RetentionManager started. Enforcing compliance policies.")
        
    def _retention_worker(self):
        while self.running:
            try:
                self.enforce_policies()
            except Exception as e:
                logger.error(f"Retention enforcement failed: {e}")
                
            # Sleep for configured interval
            time.sleep(self.run_interval_hours * 3600)
            
    def enforce_policies(self):
        """Executes deletion of records that exceed their retention policy."""
        logger.info("Running routine data retention sweep...")
        
        for table, days in self.policies.items():
            try:
                # We specifically use SQLite date functions to purge old rows
                query = f"DELETE FROM {table} WHERE timestamp < datetime('now', '-{days} days')"
                self.db.execute(query)
                
                # In a full enterprise system, we would also query the rows first to delete 
                # associated files (snapshots/videos) from the local disk using pathlib.
            except Exception as e:
                logger.warning(f"Failed to enforce retention on {table}: {e}")
                
        # Run SQLite VACUUM to reclaim disk space after large deletions
        try:
            self.db.execute("VACUUM")
        except Exception:
            pass
