from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse
from typing import List, Optional
from datetime import datetime
import os
from api.schemas.incident import IncidentSummary, IncidentDetail
from api.schemas.common import PaginatedResponse, Pagination

router = APIRouter()

@router.get("/", response_model=PaginatedResponse)
async def list_incidents(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    camera_id: Optional[str] = None,
    severity: Optional[str] = None
):
    """Fetch paginated incidents with optional filtering."""
    # In a real implementation, this would query the DatabaseManager
    # with LIMIT and OFFSET, and return real data.
    dummy_data = [
        IncidentSummary(
            id=1,
            camera_id="cam_01",
            threat_type="FIGHT",
            severity="HIGH",
            confidence=0.88,
            timestamp=datetime.now()
        )
    ]
    pagination = Pagination(total=1, page=page, size=size, pages=1)
    return PaginatedResponse(data=dummy_data, pagination=pagination)

@router.get("/{incident_id}", response_model=IncidentDetail)
async def get_incident(incident_id: int):
    """Fetch details for a specific incident."""
    # Dummy implementation
    return IncidentDetail(
        id=incident_id,
        camera_id="cam_01",
        threat_type="FIGHT",
        severity="HIGH",
        confidence=0.88,
        timestamp=datetime.now(),
        description="Fight detected in lobby",
        snapshot_path="/app/snapshots/test.jpg"
    )

@router.get("/{incident_id}/video")
async def get_incident_video(incident_id: int):
    """Stream the 15-second MP4 evidence buffer for the incident."""
    # In a real implementation, we would query the database for the video_evidence_path.
    # For now, we will simulate this by checking if a dummy video file exists.
    # We will assume recordings are saved in the 'recordings' directory.
    from utils.runtime_isolation import RuntimePath
    
    # Ideally query DB here: video_path = db.execute("SELECT video_evidence_path FROM incidents WHERE id=?", (incident_id,)).fetchone()[0]
    
    # Since this is a placeholder response for the API layer:
    recordings_dir = RuntimePath.RECORDINGS
    
    # Find any video for this incident ID in the recordings directory
    if recordings_dir.exists():
        for file in os.listdir(recordings_dir):
            if file.startswith(f"incident_{incident_id}_") and file.endswith(".mp4"):
                video_path = recordings_dir / file
                return FileResponse(path=video_path, media_type="video/mp4")
                
    raise HTTPException(status_code=404, detail="Video evidence not found for this incident")
