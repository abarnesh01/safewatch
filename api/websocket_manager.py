import asyncio
import json
from typing import List, Dict, Any
from fastapi import WebSocket
from loguru import logger

class WebSocketManager:
    """Manages real-time WebSocket connections and event broadcasting."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total clients: {len(self.active_connections)}")
        
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total clients: {len(self.active_connections)}")
        
    async def broadcast(self, event_type: str, payload: Dict[str, Any]):
        """Broadcasts an event to all connected clients."""
        message = json.dumps({"type": event_type, "payload": payload})
        
        async with self._lock:
            for connection in list(self.active_connections):
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.debug(f"Failed to send to websocket: {e}")
                    self.active_connections.remove(connection)

# Singleton instance
ws_manager = WebSocketManager()
