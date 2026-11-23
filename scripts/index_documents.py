"""Build/update the Qdrant dense index from ingested child chunks.

Run from the repository root: python -m scripts.index_documents
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.retrieval.child_store import QdrantChildStore
from app.retrieval.embeddings import get_embeddings


def main() -> None:
    child_path = Path("data/child_chunks.json")
    if not child_path.exists():
        raise SystemExit(
            "data/child_chunks.json not found. Run "
            "`python -m app.ingestion.ingest` first."
        )
    chunks = json.loads(child_path.read_text(encoding="utf-8"))
    if not chunks:
        print("No child chunks found; nothing to index.")
        return

    vectors = get_embeddings().embed_documents([item["text"] for item in chunks])
    store = QdrantChildStore()
    count = store.upsert(chunks, vectors)
    print(json.dumps({
        "chunks_indexed": count,
        "collection": store.collection_name,
        "embedding_model": "gemini-embedding-001",
        "qdrant_mode": "remote" if os.getenv("QDRANT_URL") else "local persistent",
    }, indent=2))


if __name__ == "__main__":
    main()
