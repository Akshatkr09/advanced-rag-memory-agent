from fastapi import APIRouter

from app.schemas.request_models import ChatRequest

router = APIRouter(prefix="/api", tags=["api"])


@router.post("/chat")
def chat(payload: ChatRequest):
    # Implemented in a later milestone when the RAG pipeline is connected.
    return {
        "message": "Chat pipeline is not connected yet.",
        "query": payload.query,
        "user_id": payload.user_id,
        "session_id": payload.session_id,
        "telemetry": {
            "retrieval_latency_ms": 0.0,
            "rerank_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        },
    }


@router.get("/users/{user_id}/memories")
def get_user_memories(user_id: str):
    # Replaced with the persistent Mem0 integration in the memory milestone.
    return {"user_id": user_id, "memories": []}
