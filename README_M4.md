# Milestone 4 — Cohere Reranking and Context Compression

M4 adds Cohere ReRank to the existing Gemini + Qdrant + BM25 + RRF pipeline.

## Flow
1. Hybrid retrieval fetches up to 20 child-chunk candidates.
2. `HybridSearchRetriever` adapts the custom retriever to LangChain's `BaseRetriever`.
3. `ContextualCompressionRetriever` applies Cohere's `CohereRerank` compressor.
4. Relevance scores are threshold-filtered and the top 5 are retained.
5. Selected child chunks are expanded to their parent passages.

## Configure Cohere
Add this to the existing `.env` file:

```env
COHERE_API_KEY=your_cohere_api_key
RERANK_MODEL=rerank-english-v3.0
RERANK_TOP_N=5
RERANK_MIN_SCORE=0.0
```

Do not commit `.env` or share your API key. The test suite uses a fake compressor and does not call Cohere.

## Install dependencies
```powershell
python -m pip install -r requirements.txt
```

## Run tests
```powershell
python -m pytest -v
```

## Try real reranking (after setting COHERE_API_KEY)
```powershell
python -c "from app.retrieval.hybrid_retriever import HybridRetriever; from app.retrieval.reranker import rerank_hybrid_search; r=HybridRetriever(); x=rerank_hybrid_search('What is Python?', r, fetch_k=20, top_n=5); print('Candidates:',x['candidates_retrieved']); print('Reranked matches:',len(x['matches'])); print('Parent contexts:',len(x['parents'])); [print(m.get('rerank_score'),m.get('text','')[:120]) for m in x['matches']]"
```

The real call consumes Cohere API usage. Adjust `RERANK_MIN_SCORE` based on observed scores and evaluation; zero is the initial permissive default, not a universal quality threshold.
