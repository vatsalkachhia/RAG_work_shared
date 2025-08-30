from fastapi import FastAPI, Header
from pydantic import BaseModel
import uuid
import asyncio
import time
from typing import Any, Dict
import contextlib

app = FastAPI()

# In-memory storage for demo purposes
# Each entry structure: { "history": list[str], "last_access_ts": float }
user_data_store: Dict[str, Dict[str, Any]] = {}

SESSION_TTL_SECONDS = 60 * 60  # 1 hour
CLEANUP_INTERVAL_SECONDS = 5 * 60  # 5 minutes

_cleanup_task: asyncio.Task | None = None

async def _cleanup_expired_sessions() -> None:
    while True:
        now = time.time()
        expired_user_ids = [
            user_id
            for user_id, data in list(user_data_store.items())
            if now - data.get("last_access_ts", now) > SESSION_TTL_SECONDS
        ]
        for user_id in expired_user_ids:
            user_data_store.pop(user_id, None)
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)

@app.on_event("startup")
async def _on_startup() -> None:
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_expired_sessions())

@app.on_event("shutdown")
async def _on_shutdown() -> None:
    global _cleanup_task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _cleanup_task
        _cleanup_task = None

class SessionRequest(BaseModel):
    data: str

@app.post("/session")
def session_endpoint(request: SessionRequest, x_user_id: str = Header(None)):
    if not x_user_id:
        # No user ID — create a new session
        new_user_id = str(uuid.uuid4())
        user_data_store[new_user_id] = {
            "history": [request.data],
            "last_access_ts": time.time(),
        }
        return {
            "userId": new_user_id,
            "message": "New session started",
            "userData": user_data_store[new_user_id]
        }
    
    # If user ID is given
    if x_user_id not in user_data_store:
        return {"error": "Invalid user ID"}
    
    # Append new data to the session
    user_data_store[x_user_id]["history"].append(request.data)
    user_data_store[x_user_id]["last_access_ts"] = time.time()
    
    return {
        "userId": x_user_id,
        "message": "Session updated",
        "userData": user_data_store[x_user_id]
    }
