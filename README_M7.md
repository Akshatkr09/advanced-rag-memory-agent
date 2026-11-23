# M7 — Reliability, error handling, and user isolation

M7 strengthens the M6 orchestration layer without changing the selected stack:
Gemini for generation and embeddings, Cohere for reranking, Qdrant for vector
storage, Mem0 OSS for semantic memory, and FastAPI for the API.

## Changes

- Converts retrieval, memory lookup, reranking, parent-context, generation, and
  memory-persistence failures into controlled `RAGServiceError` exceptions.
- Keeps a missing local document index as a supported no-document mode.
- Validates query, user ID, and session ID before processing.
- Rejects empty generation output and avoids saving an invalid answer.
- Adds tests proving that one user's memory context is not injected into another
  user's prompt.
- Adds API tests confirming dependency failures return HTTP 503.
- Preserves the four positive latency telemetry fields.

## Apply

Copy the ZIP contents over the existing project. Do not replace `.env`, `.venv`,
`.git`, or local `data/` files.

Install/check dependencies if needed:

```powershell
python -m pip install -r requirements.txt
python -m pytest -v
```

The new reliability tests use fakes; they do not consume Gemini or Cohere quota.
The live integrations already verified in M6 remain the final end-to-end check.

## Expected new tests

- `tests/test_m7_reliability.py`
- `tests/test_api_m7_errors.py`

Do not commit M7 until the complete test suite passes.
