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

@router.get("/status")
async def get_system_status():
    """Returns detailed system operational status."""
    # In a real scenario, this would query the DB manager and observability engine
    return {"status": "operational", "components": {"database": "ok", "camera_stream": "ok"}}
