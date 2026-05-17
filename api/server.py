from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

from api.routes import system, incidents, cameras
from api.websocket_manager import ws_manager

app = FastAPI(
    title="SafeWatch API",
    description="REST API and WebSocket event stream for SafeWatch platform.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(system.router, prefix="/api/system", tags=["System"])
app.include_router(incidents.router, prefix="/api/incidents", tags=["Incidents"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["Cameras"])

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time event stream for dashboards and mobile clients."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages if needed
            data = await websocket.receive_text()
            logger.debug(f"Received WS message: {data}")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect(websocket)

def start_server(host="0.0.0.0", port=8000):
    """Programmatic entry point for starting the FastAPI server."""
    logger.info(f"Starting SafeWatch API Server on {host}:{port}")
    uvicorn.run("api.server:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_server()
