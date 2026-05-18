import psutil
import time
from fastapi.responses import PlainTextResponse

class MetricsExporter:
    """Generates Prometheus-compatible plain-text metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process()
        
    def generate_metrics(self, fps: float, total_cameras: int, active_incidents: int) -> str:
        """Builds the prometheus format string."""
        metrics = []
        
        # System Hardware
        cpu_usage = psutil.cpu_percent()
        ram_usage = self.process.memory_info().rss / (1024 * 1024) # MB
        uptime = time.time() - self.start_time
        
        metrics.append("# HELP safewatch_cpu_percent CPU Usage percentage")
        metrics.append("# TYPE safewatch_cpu_percent gauge")
        metrics.append(f"safewatch_cpu_percent {cpu_usage}")
        
        metrics.append("# HELP safewatch_ram_mb RAM Usage in MB")
        metrics.append("# TYPE safewatch_ram_mb gauge")
        metrics.append(f"safewatch_ram_mb {ram_usage}")
        
        metrics.append("# HELP safewatch_uptime_seconds Process uptime")
        metrics.append("# TYPE safewatch_uptime_seconds counter")
        metrics.append(f"safewatch_uptime_seconds {uptime}")
        
        # Application Logic
        metrics.append("# HELP safewatch_pipeline_fps AI Inference FPS")
        metrics.append("# TYPE safewatch_pipeline_fps gauge")
        metrics.append(f"safewatch_pipeline_fps {fps}")
        
        metrics.append("# HELP safewatch_active_cameras Number of connected RTSP streams")
        metrics.append("# TYPE safewatch_active_cameras gauge")
        metrics.append(f"safewatch_active_cameras {total_cameras}")
        
        metrics.append("# HELP safewatch_total_incidents Total logged incidents")
        metrics.append("# TYPE safewatch_total_incidents counter")
        metrics.append(f"safewatch_total_incidents {active_incidents}")
        
        return "\n".join(metrics) + "\n"
