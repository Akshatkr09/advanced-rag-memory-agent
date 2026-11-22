"""Hybrid dense + BM25 retrieval with RRF and parent-document expansion."""
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from app.ingestion.parent_store import ParentDocumentStore
from app.retrieval.child_store import QdrantChildStore
from app.retrieval.embeddings import get_embeddings
from app.retrieval.rrf import reciprocal_rank_fusion

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())

class HybridRetriever:
    def __init__(
        self,
        child_path="data/child_chunks.json",
        parent_path="data/parent_documents.json",
        qdrant_store=None,
        embeddings=None,
        rrf_k=60,
    ):
        load_dotenv()
        self.child_path = Path(child_path)
        if not self.child_path.exists():
            raise FileNotFoundError(
                f"{self.child_path} not found. Run document ingestion first."
            )
        self.children = json.loads(self.child_path.read_text(encoding="utf-8"))
        self.parents = ParentDocumentStore(parent_path)
        self.qdrant = qdrant_store or QdrantChildStore()
        self.embeddings = embeddings or get_embeddings()
        self.rrf_k = rrf_k
        self._bm25 = BM25Okapi([tokenize(item["text"]) for item in self.children]) if self.children else None

    def _sparse_search(self, query, limit):
        if not self.children or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for index in ranked_indices:
            if len(results) >= limit:
                break
            item = dict(self.children[index])
            item["sparse_score"] = float(scores[index])
            results.append(item)
        return results

    def search(self, query: str, limit=20, dense_limit=20, sparse_limit=20):
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        if not self.children:
            return {"matches": [], "parents": []}
        query_vector = self.embeddings.embed_query(query)
        dense_results = self.qdrant.search(query_vector, limit=dense_limit)
        sparse_results = self._sparse_search(query, sparse_limit)
        fused = reciprocal_rank_fusion(
            dense_results, sparse_results, k=self.rrf_k
        )[:limit]
        parent_ids = [item["parent_id"] for item in fused if item.get("parent_id")]
        parent_docs = self.parents.get_many(parent_ids)
        return {"matches": fused, "parents": parent_docs}
