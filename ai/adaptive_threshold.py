from loguru import logger

class AdaptiveThreshold:
    """Uses rolling historical data to establish baselines and auto-adjust alarm sensitivity."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        
    def recalibrate(self, camera_id: str, base_threshold: float = 0.5) -> float:
        """
        Dynamically adjusts the required confidence for an alert based on historical false positives.
        If a camera generates too many false positives, its threshold hardens.
        """
        try:
            # Calculate false positive rate for this specific camera
            cursor = self.db.execute('''
                SELECT 
                    SUM(CASE WHEN f.feedback_type = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(f.id), 0)
                FROM operator_feedback f
                JOIN incidents i ON f.incident_id = i.id
                WHERE i.camera_id = ?
            ''', (camera_id,))
            
            result = cursor.fetchone()
            fp_rate = result[0] if result and result[0] is not None else 0.0
            
            # Auto-Adjustment Logic
            if fp_rate > 0.20:
                # Too noisy: Harden the threshold (max 0.95)
                new_threshold = min(0.95, base_threshold + (fp_rate * 0.5))
                logger.info(f"AdaptiveThreshold: {camera_id} hardened to {new_threshold:.2f} due to high FP rate.")
                return new_threshold
            elif fp_rate < 0.05 and fp_rate > 0:
                # Very accurate: Lower threshold slightly to catch subtle threats
                new_threshold = max(0.30, base_threshold - 0.05)
                logger.debug(f"AdaptiveThreshold: {camera_id} softened to {new_threshold:.2f}")
                return new_threshold
                
            return base_threshold
        except Exception as e:
            logger.error(f"Failed to recalibrate threshold for {camera_id}: {e}")
            return base_threshold
