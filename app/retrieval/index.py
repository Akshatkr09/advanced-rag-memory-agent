"""Build/update the local or remote Qdrant child-chunk index.

git commit -m "M4: Add Cohere reranking and context compression"Run after ingestion:
    python -m app.ingestion.ingest
    python -m app.retrieval.index
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from app.retrieval.child_store import QdrantChildStore
from app.retrieval.embeddings import get_embeddings

def index_child_chunks(child_path="data/child_chunks.json", batch_size=64):
    load_dotenv()
    path = Path(child_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run `python -m app.ingestion.ingest` first."
        )
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not chunks:
        return {"chunks_indexed": 0, "message": "No child chunks to index."}

    embeddings = get_embeddings()
    store = QdrantChildStore()
    indexed = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = embeddings.embed_documents([item["text"] for item in batch])
        indexed += store.upsert(batch, vectors)
    return {
        "chunks_indexed": indexed,
        "collection": store.collection_name,
        "embedding_model": "gemini-embedding-001",
        "qdrant_mode": "remote" if os.getenv("QDRANT_URL") else "local persistent",
    }

if __name__ == "__main__":
    print(json.dumps(index_child_chunks(), indent=2))
