import os
from pathlib import Path
from loguru import logger

class RuntimePath:
    """Manages isolated runtime directories for SafeWatch."""
    
    ROOT = Path(__file__).parent.parent
    
    # Isolated Runtime Dirs
    CACHE = ROOT / "runtime_cache"
    TELEMETRY = ROOT / "telemetry_cache"
    SNAPSHOTS = ROOT / "snapshots"
    LOGS = ROOT / "logs"
    EXPORTS = ROOT / "exports"
    RECORDINGS = ROOT / "recordings"

    @classmethod
    def ensure_isolation(cls):
        """Ensure all runtime directories exist and are properly isolated."""
        dirs = [cls.CACHE, cls.TELEMETRY, cls.SNAPSHOTS, cls.LOGS, cls.EXPORTS, cls.RECORDINGS]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            # Create a local .gitignore in each if not exists
            gitignore = d / ".gitignore"
            if not gitignore.exists():
                with open(gitignore, "w") as f:
                    f.write("*\n!.gitignore\n")
        
        logger.info("Runtime isolation layer activated.")

    @classmethod
    def get_cache_file(cls, filename: str) -> Path:
        return cls.CACHE / filename

    @classmethod
    def get_telemetry_file(cls, filename: str) -> Path:
        return cls.TELEMETRY / filename
