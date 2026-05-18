from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    version: str

@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns the basic health status of the API."""
    return {"status": "ok", "version": "1.0.0"}

@router.get("/metrics")
async def get_metrics():
    """Returns Prometheus-compatible plain-text metrics."""
    from monitoring.metrics import MetricsExporter
    from main import app
    from fastapi.responses import PlainTextResponse
    
    exporter = MetricsExporter()
    
    # We grab live stats from the global app context
    fps = 0.0
    total_cameras = 0
    total_incidents = 0
    
    try:
        # Mocking or extracting real values
        if hasattr(app, '_db_manager'):
            cursor = app._db_manager.execute("SELECT COUNT(id) FROM incidents")
            if cursor:
                total_incidents = cursor.fetchone()[0]
        if hasattr(app, '_stream_manager'):
            total_cameras = len(app._stream_manager.streams)
    except Exception:
        pass
        
    metrics_str = exporter.generate_metrics(fps, total_cameras, total_incidents)
    return PlainTextResponse(metrics_str)

@router.get("/status")
async def get_system_status():
    """Returns detailed system operational status."""
    # In a real scenario, this would query the DB manager and observability engine
    return {"status": "operational", "components": {"database": "ok", "camera_stream": "ok"}}
