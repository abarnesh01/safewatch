from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

class IncidentSummary(BaseModel):
    id: int
    camera_id: str
    threat_type: str
    severity: str
    confidence: float
    timestamp: datetime

class IncidentDetail(IncidentSummary):
    description: str
    snapshot_path: Optional[str] = None
    correlation_id: Optional[str] = None
    tags: Optional[str] = None
    metadata: Optional[dict] = None
