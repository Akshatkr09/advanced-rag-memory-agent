import pytest
from app.retrieval.rrf import reciprocal_rank_fusion

def test_rrf_promotes_items_present_in_both_rankings():
    dense = [{"child_id": "a"}, {"child_id": "b"}]
    sparse = [{"child_id": "b"}, {"child_id": "c"}]
    result = reciprocal_rank_fusion(dense, sparse, k=60)
    assert result[0]["child_id"] == "b"
    assert result[0]["rrf_score"] > result[1]["rrf_score"]

def test_rrf_accepts_string_ids():
    result = reciprocal_rank_fusion(["x", "y"], ["y", "z"], k=10)
    assert result[0]["child_id"] == "y"

def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(["x"], k=0)

def test_rrf_preserves_dense_and_sparse_metadata_for_same_child():
    dense = [{"child_id": "shared", "parent_id": "p1", "dense_score": 0.91}]
    sparse = [{"child_id": "shared", "parent_id": "p1", "sparse_score": 4.2}]
    result = reciprocal_rank_fusion(dense, sparse, k=60)
    assert len(result) == 1
    assert result[0]["dense_score"] == 0.91
    assert result[0]["sparse_score"] == 4.2
    assert result[0]["parent_id"] == "p1"
