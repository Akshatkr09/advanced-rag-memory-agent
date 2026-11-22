# Milestone 3 — Hybrid Retrieval

This milestone adds:
- Qdrant storage for child-chunk vectors.
- OpenAI `text-embedding-3-small` dense embeddings.
- BM25 keyword retrieval.
- Reciprocal Rank Fusion (RRF).
- Parent-document expansion after child retrieval.

## 1. Configure the API key

Copy `.env.example` to `.env` if you do not already have `.env`, then set:

```env
OPENAI_API_KEY=your_key_here
```

Do not commit `.env`. The project uses Qdrant's persistent local mode by default, so a separate Qdrant server is not required for local development. You can optionally set `QDRANT_URL` and `QDRANT_API_KEY` to use a remote Qdrant instance.

## 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 3. Ingest documents and build the vector index

Put PDF/TXT documents in `data/documents/`, then run:

```powershell
python -m app.ingestion.ingest
python -m app.retrieval.index
```

The indexing step calls the OpenAI embeddings API and may incur API usage charges.

## 4. Try hybrid retrieval

```powershell
python -c "from app.retrieval.hybrid_retriever import HybridRetriever; r=HybridRetriever(); result=r.search('What is Python?', limit=5); print(result)"
```

The result contains `matches` (fused child-chunk matches) and `parents` (larger parent passages for context).

## 5. Run tests

```powershell
python -m pytest -v
```

Tests use mocked embeddings/Qdrant responses and do not need API keys or network access.
