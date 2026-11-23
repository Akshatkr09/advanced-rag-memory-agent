# Milestone 6 — RAG + Semantic Memory + Gemini API Orchestration

M6 connects the existing retrieval and memory components to the FastAPI endpoints.

## What is connected
- `POST /api/chat` accepts `query`, `user_id`, and `session_id`.
- Hybrid retrieval searches indexed child chunks using dense Gemini embeddings and BM25, then fuses results with RRF.
- Cohere reranks fused candidates and keeps up to five relevant child chunks.
- Selected child chunks expand to larger parent-document context.
- Mem0 retrieves the user's relevant long-term facts before generation.
- Gemini generates the answer using retrieved knowledge and the remembered profile.
- The user/assistant exchange is saved to Mem0 using the supplied user and session IDs.
- `GET /api/users/{user_id}/memories` returns stored facts.
- Chat responses include retrieval, reranking, generation, and total latency in milliseconds.

## Environment
Keep the existing keys in `.env` and set:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
MEM0_LLM_MODEL=gemini-3.5-flash-lite
MEM0_EMBEDDING_MODEL=models/gemini-embedding-001
MEM0_EMBEDDING_DIMS=768
COHERE_API_KEY=your_cohere_api_key
RERANK_MODEL=rerank-english-v3.0
RERANK_TOP_N=5
RERANK_MIN_SCORE=0.0
```

The Gemini model is configurable. Free-tier quotas and availability are controlled by Google.

## Start the API
```powershell
uvicorn app.main:app --reload
```
Open `http://127.0.0.1:8000/docs`.

## Chat request
`POST http://127.0.0.1:8000/api/chat`

```json
{
  "query": "Write a quick merge sort routine.",
  "user_id": "usr_99",
  "session_id": "session_2"
}
```

Response includes `answer`, `user_id`, `session_id`, `sources`, and a `telemetry` object with four numeric millisecond values. Latencies vary by API response time and local hardware.

## Memory endpoint
`GET http://127.0.0.1:8000/api/users/usr_99/memories`

## Knowledge-base setup
Place `.txt` or `.pdf` files under `data/documents/`, then run:
```powershell
python -m app.ingestion.ingest
python -m app.retrieval.index
```
The chat endpoint can answer without an indexed knowledge base, but document-grounded retrieval requires ingestion and indexing.

## Tests
```powershell
python -m pytest -v
```
M6 tests use fakes for Gemini, Cohere, retrieval, and Mem0; they do not spend API quota.
