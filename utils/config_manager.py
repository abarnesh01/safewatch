import os
import yaml
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ConfigManager:
    """Manages system configuration, merging YAML and environment variables."""

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self._config = self._load_yaml()
        self._apply_env_overrides()
        self._validate_security()

    def _load_yaml(self) -> dict:
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return {}
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config.yaml: {e}")
            return {}

    def _apply_env_overrides(self):
        """Overrides config keys with environment variables."""
        if "system" not in self._config:
            self._config["system"] = {}
        if "security" not in self._config:
            self._config["security"] = {}
        
        # System
        env_val = os.getenv("SAFEWATCH_ENV")
        if env_val:
            self._config["system"]["env"] = env_val
        
        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            self._config["system"]["log_level"] = log_level

        gpu_accel = os.getenv("ENABLE_GPU_ACCELERATION")
        if gpu_accel is not None:
            self._config["system"]["device"] = "cuda" if gpu_accel.lower() == "true" else "cpu"

        # Security
        secret_key = os.getenv("SAFEWATCH_SECRET_KEY")
        if secret_key:
            self._config["security"]["secret_key"] = secret_key

        session_timeout = os.getenv("SESSION_TIMEOUT_MINUTES")
        if session_timeout:
            self._config["security"]["session_timeout_minutes"] = int(session_timeout)

        admin_user = os.getenv("DEFAULT_ADMIN_USER")
        if admin_user:
            self._config["security"]["default_admin_user"] = admin_user

        admin_pass = os.getenv("DEFAULT_ADMIN_PASS")
        if admin_pass:
            self._config["security"]["default_admin_pass"] = admin_pass

    def _validate_security(self):
        """Validates critical security settings for production."""
        env = self._config.get("system", {}).get("env", "development")
        secret_key = self._config.get("security", {}).get("secret_key")
        
        if env == "production":
            if not secret_key or secret_key == "your_secure_random_secret_key_here":
                logger.warning("INSECURE: Using default/missing SAFEWATCH_SECRET_KEY in production!")
            
            admin_pass = self._config.get("security", {}).get("default_admin_pass")
            if not admin_pass or admin_pass == "safewatch_admin_2026":
                 logger.warning("INSECURE: Default admin password is used in production. Please change it via .env")

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    def get_full_config(self) -> dict:
        return self._config
