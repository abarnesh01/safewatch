from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from datetime import datetime
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
