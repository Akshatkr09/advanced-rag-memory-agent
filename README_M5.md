# Milestone 5 — Long-term Semantic Memory with Mem0

This milestone uses **Mem0 OSS (self-hosted)** rather than the hosted Mem0 Platform. It does not require a separate `MEM0_API_KEY`. The existing Gemini API key powers both Mem0's memory extraction LLM and its embeddings. Mem0 stores memory vectors in a separate persistent local Qdrant collection, isolated from the document-retrieval collection.

## Configuration
The project `.env` must contain the Gemini key already used in M3:

```env
GEMINI_API_KEY=your_existing_gemini_key
MEM0_LLM_MODEL=gemini-2.0-flash
MEM0_EMBEDDING_MODEL=models/gemini-embedding-001
MEM0_EMBEDDING_DIMS=768
MEM0_QDRANT_PATH=data/mem0_qdrant
MEM0_COLLECTION=personal_memories
MEM0_HISTORY_DB_PATH=data/mem0_history.db
```

Do not commit `.env`. No Mem0 Platform key is needed for this local OSS configuration.

## Install
```powershell
python -m pip install -r requirements.txt
```

## Cross-session demonstration
```powershell
python -m app.memory.memory_demo
```

The demo uses user ID `usr_99`. It stores the sample preference in `session_1`, then searches the same user's memories using a different query as a later session would. Mem0 performs fact extraction (`infer=True`) and semantic search.

## Run tests
```powershell
python -m pytest -v
```

Tests mock the Mem0 client and do not call Gemini or Qdrant. The first real demo run uses Gemini API calls and creates local memory data under `data/`.

## Important
Mem0's local Qdrant database is separate from the project's document Qdrant database. Do not point both clients at the same local storage directory. `on_disk=True` is set explicitly so local vector data is persisted between runs.
