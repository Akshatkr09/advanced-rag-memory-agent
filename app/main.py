from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Advanced RAG Memory Agent",
    description="Hybrid retrieval, Cohere reranking, and persistent semantic memory.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "project": "advanced-rag-memory-agent"}
