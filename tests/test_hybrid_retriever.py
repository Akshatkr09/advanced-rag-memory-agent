import json
from app.retrieval.hybrid_retriever import HybridRetriever

class FakeEmbeddings:
    def embed_query(self, query):
        return [0.1, 0.2, 0.3]

class FakeQdrant:
    def search(self, query_vector, limit=20):
        return [{"child_id": "c1", "parent_id": "p1", "text": "dense match"}]

def test_hybrid_search_fuses_results_and_returns_parent(tmp_path):
    child_path = tmp_path / "children.json"
    parent_path = tmp_path / "parents.json"
    children = [
        {"child_id": "c1", "parent_id": "p1", "document_id": "d1",
         "source": "test.txt", "text": "Python functions and programming"},
        {"child_id": "c2", "parent_id": "p2", "document_id": "d1",
         "source": "test.txt", "text": "Cooking pasta with tomato sauce"},
    ]
    parents = [
        {"parent_id": "p1", "document_id": "d1", "source": "test.txt",
         "text": "A larger parent passage about Python functions and programming."},
        {"parent_id": "p2", "document_id": "d1", "source": "test.txt",
         "text": "A larger parent passage about cooking pasta."},
    ]
    child_path.write_text(json.dumps(children), encoding="utf-8")
    parent_path.write_text(json.dumps(parents), encoding="utf-8")
    retriever = HybridRetriever(
        child_path=child_path,
        parent_path=parent_path,
        qdrant_store=FakeQdrant(),
        embeddings=FakeEmbeddings(),
    )
    result = retriever.search("Python programming")
    assert result["matches"]
    assert result["matches"][0]["child_id"] == "c1"
    assert result["parents"][0]["parent_id"] == "p1"
    assert "larger parent passage" in result["parents"][0]["text"]
