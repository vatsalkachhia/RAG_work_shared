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


# =====================================================
# NEW CODE STARTS HERE
# =====================================================
from fastapi import UploadFile, File, HTTPException
from engine import RAGEngine  # Import your RAG pipeline
from typing import Optional

# Store per-user RAG engine
rag_sessions: Dict[str, RAGEngine] = {}

# Default config (you can tune if needed)
DEFAULT_RAG_CONFIG = {
    "chunking": "recursive",
    "embedding": "huggingface",
    "vectordb": "faiss",
    "retrieval": "topk",
    "llm": "groq",
    "memory": "windowed",
    "reranker": False,
}

class ChatRequest(BaseModel):
    question: str

class UploadTextRequest(BaseModel):
    text: str


def _touch_session(user_id: str):
    """Ensure session is valid and refresh timestamp"""
    if user_id not in user_data_store:
        raise HTTPException(status_code=401, detail="Invalid or expired user ID")
    user_data_store[user_id]["last_access_ts"] = time.time()


def _get_or_create_rag(user_id: str) -> RAGEngine:
    """Get or create a RAG engine for this session"""
    if user_id not in rag_sessions:
        rag_sessions[user_id] = RAGEngine(DEFAULT_RAG_CONFIG.copy())
    return rag_sessions[user_id]


@app.post("/upload-text")
async def upload_text(payload: UploadTextRequest, x_user_id: Optional[str] = Header(None)):
    """Upload raw text to build knowledge base"""
    if not x_user_id:
        raise HTTPException(status_code=400, detail="x-user-id header required")
    _touch_session(x_user_id)

    rag = _get_or_create_rag(x_user_id)
    rag.build_knowledge_base(payload.text.strip())

    return {"userId": x_user_id, "message": "Knowledge base built from text"}


@app.post("/upload-file")
async def upload_file(x_user_id: Optional[str] = Header(None), file: UploadFile = File(...)):
    """Upload .txt file and build knowledge base"""
    if not x_user_id:
        raise HTTPException(status_code=400, detail="x-user-id header required")
    _touch_session(x_user_id)

    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=415, detail="Only .txt supported for now")

    text = (await file.read()).decode("utf-8")
    rag = _get_or_create_rag(x_user_id)
    rag.build_knowledge_base(text)

    return {"userId": x_user_id, "message": f"Knowledge base built from {file.filename}"}


@app.post("/chat")
async def chat_with_doc(request: ChatRequest, x_user_id: Optional[str] = Header(None)):
    """Chat with the uploaded document"""
    if not x_user_id:
        raise HTTPException(status_code=400, detail="x-user-id header required")
    _touch_session(x_user_id)

    if x_user_id not in rag_sessions or rag_sessions[x_user_id].vectorstore is None:
        raise HTTPException(status_code=409, detail="No knowledge base found. Upload a document first.")

    rag = rag_sessions[x_user_id]
    result = rag.query(request.question)

    return {
        "userId": x_user_id,
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }
# =====================================================
# NEW CODE ENDS HERE
# =====================================================
