from loguru import logger

class ModelRegistry:
    """Manages the versioning, A/B testing, and hot-swapping of ML models."""
    
    def __init__(self, db_manager):
        self.db = db_manager

    def register_model(self, name: str, version: str, path: str, accuracy: float):
        """Register a new model version into STAGING."""
        try:
            self.db.execute(
                "INSERT INTO models_registry (model_name, version, weights_path, accuracy, deployment_status) VALUES (?, ?, ?, ?, 'STAGING')",
                (name, version, path, accuracy)
            )
            logger.info(f"Registered model {name} v{version} to STAGING.")
            return True
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return False

    def get_active_model(self, name: str):
        """Retrieve the currently ACTIVE weights path for a model."""
        cursor = self.db.execute(
            "SELECT version, weights_path FROM models_registry WHERE model_name = ? AND deployment_status = 'ACTIVE'",
            (name,)
        )
        if cursor:
            return cursor.fetchone()
        return None

    def activate_model(self, model_id: int, name: str):
        """Promotes a STAGING model to ACTIVE and retires the old one."""
        try:
            # Retire old active
            self.db.execute("UPDATE models_registry SET deployment_status = 'RETIRED' WHERE model_name = ? AND deployment_status = 'ACTIVE'", (name,))
            # Activate new
            self.db.execute("UPDATE models_registry SET deployment_status = 'ACTIVE' WHERE id = ?", (model_id,))
            logger.info(f"Activated model ID {model_id} for {name}.")
            return True
        except Exception as e:
            logger.error(f"Model activation failed: {e}")
            return False
