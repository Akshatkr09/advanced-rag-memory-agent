import pytest
from app.ingestion.chunking import create_parent_child_chunks

def test_parent_child_mapping_and_sizes():
    text = " ".join(f"word{i}" for i in range(500))
    parents, children = create_parent_child_chunks(text, "sample.txt")
    assert parents and children
    assert all(len(p.text) <= 1000 for p in parents)
    assert all(len(c.text) <= 200 for c in children)
    parent_ids = {p.parent_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)

def test_empty_document():
    assert create_parent_child_chunks("  ", "empty.txt") == ([], [])

def test_invalid_overlap():
    with pytest.raises(ValueError):
        create_parent_child_chunks("hello", "x.txt", child_size=100, child_overlap=100)
