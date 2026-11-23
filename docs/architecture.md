# Architecture notes

## Request flow

1. `POST /api/chat` validates `query`, `user_id`, and `session_id`.
2. `HybridRetriever` performs dense Qdrant search and sparse BM25 search.
3. Reciprocal Rank Fusion merges the ranked lists.
4. Mem0 searches for relevant facts scoped to the supplied `user_id`.
5. Cohere reranks the child-chunk candidates; top-N and minimum-score settings are applied.
6. Selected child chunks are expanded to parent-document context.
7. Gemini generates an answer using the question, document context, and user profile.
8. The user and assistant messages are saved to Mem0 with the same user ID and session ID.
9. The response includes the answer, source paths, and stage-level latency telemetry.

## Storage boundaries

- Document knowledge base: child vectors/metadata in Qdrant collection `child_chunks`; parent text in `data/parent_documents.json`; child text/BM25 corpus in `data/child_chunks.json`.
- User memory: Mem0 OSS uses a separate Qdrant path and collection, plus a local history database. It is scoped by `user_id`.

## API error behavior

- Invalid request fields return HTTP 400.
- Controlled runtime/dependency failures return HTTP 503.
- A missing local child-chunk index is treated as no-document mode; the assistant can still answer using general knowledge and user memory.
