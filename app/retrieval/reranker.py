"""Cohere reranking and LangChain contextual-compression integration."""
import os
from typing import Any

from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.ingestion.parent_store import ParentDocumentStore


class HybridSearchRetriever(BaseRetriever):
    """Adapt the project's hybrid retriever to LangChain's BaseRetriever API."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid_retriever: Any
    fetch_k: int = 20

    def _get_relevant_documents(self, query: str, *, run_manager):
        result = self.hybrid_retriever.search(query, limit=self.fetch_k)
        documents = []
        for match in result.get("matches", []):
            metadata = {
                key: value for key, value in match.items()
                if key not in {"text", "rrf_score"}
            }
            metadata["rrf_score"] = match.get("rrf_score")
            documents.append(
                Document(page_content=match.get("text", ""), metadata=metadata)
            )
        return documents


def create_cohere_reranker(
    api_key: str | None = None,
    model: str | None = None,
    top_n: int = 5,
):
    """Create the Cohere compressor. API key is read from .env if omitted."""
    load_dotenv()
    key = api_key or os.getenv("COHERE_API_KEY")
    if not key:
        raise RuntimeError(
            "COHERE_API_KEY is missing. Add your Cohere API key to the project's .env file."
        )
    from langchain_cohere import CohereRerank

    return CohereRerank(
        cohere_api_key=key,
        model=model or os.getenv("RERANK_MODEL", "rerank-english-v3.0"),
        top_n=top_n,
    )


def rerank_candidates(
    query: str,
    candidates: list[dict],
    compressor,
    top_n: int = 5,
    min_score: float = 0.0,
) -> list[dict]:
    """Rerank candidate dictionaries, apply a score threshold, and keep top N.

    This function is deliberately injectable so unit tests can use a fake
    compressor without making Cohere API calls.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    documents = [
        Document(
            page_content=item.get("text", ""),
            metadata={key: value for key, value in item.items() if key != "text"},
        )
        for item in candidates
    ]
    if not documents:
        return []

    reranked_docs = compressor.compress_documents(documents, query)
    results = []
    for doc in reranked_docs:
        score = doc.metadata.get("relevance_score")
        if score is None:
            # Cohere normally supplies relevance_score. Missing scores are
            # treated as zero so a positive threshold cannot be bypassed.
            score = 0.0
        score = float(score)
        if score < min_score:
            continue
        item = dict(doc.metadata)
        item["text"] = doc.page_content
        item["rerank_score"] = score
        results.append(item)

    results.sort(key=lambda item: item["rerank_score"], reverse=True)
    return results[:top_n]


def rerank_hybrid_search(
    query: str,
    hybrid_retriever,
    compressor=None,
    parent_path: str = "data/parent_documents.json",
    fetch_k: int = 20,
    top_n: int = 5,
    min_score: float | None = None,
) -> dict:
    """Retrieve hybrid candidates, rerank via ContextualCompressionRetriever,
    threshold scores, then expand the selected child chunks to parent context.
    """
    load_dotenv()
    if compressor is None:
        compressor = create_cohere_reranker(top_n=top_n)

    threshold = (
        float(os.getenv("RERANK_MIN_SCORE", "0.0"))
        if min_score is None else float(min_score)
    )

    base_retriever = HybridSearchRetriever(
        hybrid_retriever=hybrid_retriever,
        fetch_k=fetch_k,
    )
    contextual_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    compressed_docs = contextual_retriever.invoke(query)
    selected = []
    for doc in compressed_docs:
        score = float(doc.metadata.get("relevance_score", 0.0))
        if score >= threshold:
            item = dict(doc.metadata)
            item["text"] = doc.page_content
            item["rerank_score"] = score
            selected.append(item)

    selected.sort(key=lambda item: item["rerank_score"], reverse=True)
    selected = selected[:top_n]

    parent_ids = [item["parent_id"] for item in selected if item.get("parent_id")]
    parents = ParentDocumentStore(parent_path).get_many(parent_ids) if parent_ids else []

    return {
        "candidates_retrieved": fetch_k,
        "matches": selected,
        "parents": parents,
        "rerank_model": os.getenv("RERANK_MODEL", "rerank-english-v3.0"),
        "rerank_top_n": top_n,
        "rerank_min_score": threshold,
    }
