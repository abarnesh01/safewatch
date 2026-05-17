from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel

router = APIRouter()

class CameraStatus(BaseModel):
    id: str
    name: str
    status: str
    fps: float
    bandwidth_mbps: float

@router.get("/", response_model=List[CameraStatus])
async def list_cameras():
    """List all configured cameras and their real-time status."""
    return [
        CameraStatus(
            id="cam_01",
            name="Front Entrance",
            status="online",
            fps=14.5,
            bandwidth_mbps=2.1
        )
    ]
