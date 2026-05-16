import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field

@dataclass
class DetectorMetrics:
    name: str
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    executions: int = 0
    errors: int = 0
    last_run: float = 0.0

class ObservabilityEngine:
    """Thread-safe metrics aggregation for AI observability."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ObservabilityEngine, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self._metrics = defaultdict(lambda: DetectorMetrics(name="unknown"))
        self._fps_history = deque(maxlen=50)
        self._processing_breakdown = defaultdict(lambda: deque(maxlen=50))
        self._lock = threading.Lock()
        self._initialized = True

    def record_latency(self, name: str, latency_ms: float):
        with self._lock:
            m = self._metrics[name]
            m.name = name
            m.latencies.append(latency_ms)
            m.executions += 1
            m.last_run = time.time()

    def record_error(self, name: str):
        with self._lock:
            self._metrics[name].errors += 1

    def record_breakdown(self, stage: str, duration_ms: float):
        with self._lock:
            self._processing_breakdown[stage].append(duration_ms)

    def record_fps(self, fps: float):
        with self._lock:
            self._fps_history.append(fps)

    def get_summary(self):
        with self._lock:
            summary = {
                "detectors": {},
                "fps": sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0,
                "breakdown": {s: sum(d)/len(d) for s, d in self._processing_breakdown.items() if d}
            }
            for name, m in self._metrics.items():
                lats = list(m.latencies)
                summary["detectors"][name] = {
                    "avg_lat": sum(lats)/len(lats) if lats else 0,
                    "executions": m.executions,
                    "errors": m.errors,
                    "health": max(0, 100 - (m.errors * 10)) # Simple health score
                }
            return summary
