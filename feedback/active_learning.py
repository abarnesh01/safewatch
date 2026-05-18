import os
import shutil
from pathlib import Path
from loguru import logger

class ActiveLearningMiner:
    """Mines operator feedback to curate a retraining dataset of hard edge-cases."""
    
    def __init__(self, db_manager, output_dir: str = "retraining_dataset"):
        self.db = db_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def mine_false_positives(self):
        """Scans for FALSE_POSITIVE feedback and copies the snapshots to the retraining directory."""
        logger.info("Starting Active Learning mining sequence...")
        try:
            # Join feedback with incidents to get the snapshot paths
            cursor = self.db.execute('''
                SELECT f.incident_id, i.snapshot_path, f.corrected_label 
                FROM operator_feedback f
                JOIN incidents i ON f.incident_id = i.id
                WHERE f.feedback_type = 'FALSE_POSITIVE'
            ''')
            
            count = 0
            for row in cursor.fetchall():
                incident_id, snapshot_path, corrected_label = row
                if snapshot_path and os.path.exists(snapshot_path):
                    ext = Path(snapshot_path).suffix
                    label_str = corrected_label.replace(" ", "_").lower() if corrected_label else "unknown"
                    new_filename = f"fp_{incident_id}_{label_str}{ext}"
                    new_path = self.output_dir / new_filename
                    
                    if not new_path.exists():
                        shutil.copy2(snapshot_path, new_path)
                        count += 1
                        
            logger.info(f"Active Learning: Successfully mined {count} new hard-negatives for retraining.")
            return count
        except Exception as e:
            logger.error(f"Active Learning mining failed: {e}")
            return 0
