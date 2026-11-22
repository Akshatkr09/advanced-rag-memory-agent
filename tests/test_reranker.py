from langchain_core.documents import Document
import pytest

from app.retrieval.reranker import rerank_candidates


class FakeCohereCompressor:
    def __init__(self, scores):
        self.scores = scores
        self.received_query = None
        self.received_documents = None

    def compress_documents(self, documents, query):
        self.received_query = query
        self.received_documents = documents
        ranked = []
        for doc, score in zip(documents, self.scores):
            metadata = dict(doc.metadata)
            metadata["relevance_score"] = score
            ranked.append(Document(page_content=doc.page_content, metadata=metadata))
        return sorted(ranked, key=lambda d: d.metadata["relevance_score"], reverse=True)


def test_rerank_keeps_top_five_in_relevance_order():
    candidates = [
        {"child_id": f"c{i}", "parent_id": f"p{i}", "text": f"chunk {i}"}
        for i in range(8)
    ]
    scores = [0.21, 0.92, 0.41, 0.88, 0.73, 0.51, 0.99, 0.62]
    compressor = FakeCohereCompressor(scores)

    result = rerank_candidates(
        "explain Python", candidates, compressor, top_n=5, min_score=0.0
    )

    assert len(result) == 5
    assert [item["rerank_score"] for item in result] == sorted(
        [item["rerank_score"] for item in result], reverse=True
    )
    assert result[0]["child_id"] == "c6"
    assert compressor.received_query == "explain Python"
    assert len(compressor.received_documents) == 8


def test_rerank_filters_below_threshold():
    candidates = [
        {"child_id": "low", "text": "low relevance"},
        {"child_id": "mid", "text": "medium relevance"},
        {"child_id": "high", "text": "high relevance"},
    ]
    compressor = FakeCohereCompressor([0.19, 0.64, 0.91])

    result = rerank_candidates(
        "query", candidates, compressor, top_n=5, min_score=0.5
    )

    assert [item["child_id"] for item in result] == ["high", "mid"]
    assert all(item["rerank_score"] >= 0.5 for item in result)


def test_rerank_empty_candidates_returns_empty():
    result = rerank_candidates(
        "query", [], FakeCohereCompressor([]), top_n=5, min_score=0.0
    )
    assert result == []


def test_rerank_rejects_empty_query():
    with pytest.raises(ValueError):
        rerank_candidates(" ", [{"text": "x"}], FakeCohereCompressor([0.8]))


def test_rerank_rejects_invalid_top_n():
    with pytest.raises(ValueError):
        rerank_candidates("query", [], FakeCohereCompressor([]), top_n=0)
