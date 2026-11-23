from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.request_models import ChatRequest
from app.services.rag_service import RAGService

router = APIRouter(prefix="/api", tags=["api"])

@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()

@router.post("/chat")
def chat(payload: ChatRequest, service: RAGService = Depends(get_rag_service)):
    try:
        return service.chat(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@router.get("/users/{user_id}/memories")
def get_user_memories(user_id: str, service: RAGService = Depends(get_rag_service)):
    try:
        return {"user_id": user_id, "memories": service.get_user_memories(user_id)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
