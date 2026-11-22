# Milestone 3 — Hybrid Retrieval (Gemini edition)

This is the corrected M3 implementation. It uses Gemini for embeddings; it does not require an OpenAI API key.

## Stack
- Gemini Developer API: `gemini-embedding-001` embeddings
- Qdrant: persistent local vector store by default; optional remote Qdrant
- BM25: keyword retrieval
- RRF: merge dense and sparse ranked results
- Parent expansion: return parent passages for retrieved child chunks

## Configure Gemini
1. Create or open the `.env` file in the project root.
2. Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Do not share the key or commit `.env`. `GOOGLE_API_KEY` is also accepted as an alternative environment variable.

## Install
```powershell
python -m pip install -r requirements.txt
```

## Index documents
Run ingestion first if you have not already done so:

```powershell
python -m app.ingestion.ingest
python -m app.retrieval.index
```

Indexing calls the Gemini embedding API. The first run creates the Qdrant collection using the embedding vector size returned by Gemini. Keep the same embedding model for indexing and querying.

## Test hybrid retrieval
```powershell
python -c "from app.retrieval.hybrid_retriever import HybridRetriever; r=HybridRetriever(); result=r.search('What is Python?', limit=5); print(result)"
```

The result contains `matches` (fused child-chunk matches) and `parents` (larger parent passages).

## Run tests
```powershell
python -m pytest -v
```

The unit tests mock embeddings and Qdrant, so they do not require API calls.
