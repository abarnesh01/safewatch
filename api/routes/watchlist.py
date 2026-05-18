from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List
from pydantic import BaseModel
import cv2
import numpy as np

router = APIRouter()

class WatchlistEntry(BaseModel):
    id: int
    name: str
    category: str

@router.get("/", response_model=List[WatchlistEntry])
async def get_watchlist():
    """Get all enrolled persons."""
    from database.db_manager import DatabaseManager
    from recognition.watchlist_manager import WatchlistManager
    
    db = DatabaseManager()
    manager = WatchlistManager(db)
    return manager.get_watchlist()

@router.post("/enroll")
async def enroll_face(
    name: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...)
):
    """Enroll a new face via uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    from recognition.face_detector import FaceRecognitionSystem
    # Note: In production this would be loaded globally on app startup
    recognizer = FaceRecognitionSystem(use_gpu=False)
    faces = recognizer.detect_and_embed(frame)
    
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in image")
        
    embedding = faces[0].embedding
    
    from database.db_manager import DatabaseManager
    from recognition.watchlist_manager import WatchlistManager
    db = DatabaseManager()
    manager = WatchlistManager(db)
    
    success = manager.enroll_face(name, category, embedding)
    
    if success:
        return {"status": "success", "message": f"{name} enrolled as {category}."}
    raise HTTPException(status_code=500, detail="Database failure")
