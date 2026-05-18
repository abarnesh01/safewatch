import time
import requests
import threading
from loguru import logger

class EdgeSyncWorker:
    """Manages the background queuing and offline-safe syncing of edge metadata to the cloud."""
    
    def __init__(self, db_manager, cloud_url: str):
        self.db = db_manager
        self.cloud_url = cloud_url
        self.running = False
        self._thread = None
        
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()
        logger.info(f"EdgeSyncWorker started. Syncing to {self.cloud_url}")
        
    def stop(self):
        self.running = False
        
    def _sync_loop(self):
        while self.running:
            try:
                # 1. Fetch pending incidents
                # Note: Requires db migration for sync_status column
                cursor = self.db.execute("SELECT * FROM incidents WHERE sync_status='PENDING' LIMIT 50")
                if cursor:
                    pending = cursor.fetchall()
                    if pending:
                        payload = [dict(r) for r in pending]
                        
                        # 2. Transmit metadata (No Raw Video)
                        try:
                            response = requests.post(f"{self.cloud_url}/api/hub/ingest", json=payload, timeout=5)
                            
                            if response.status_code == 200:
                                ids = [p["id"] for p in payload]
                                # 3. Mark as SYNCED
                                self.db.execute(f"UPDATE incidents SET sync_status='SYNCED' WHERE id IN ({','.join(map(str, ids))})")
                                logger.info(f"Synced {len(ids)} incidents to Central Hub.")
                        except requests.exceptions.RequestException:
                            logger.warning("Cloud Hub unreachable. Queuing incidents for later sync.")
            except Exception as e:
                logger.error(f"Edge sync failed: {e}")
            
            time.sleep(10) # Polling interval
