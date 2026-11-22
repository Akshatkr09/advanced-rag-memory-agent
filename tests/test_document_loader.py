import pytest
from app.ingestion.document_loader import load_document, discover_documents

def test_load_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("RAG retrieves relevant text.", encoding="utf-8")
    assert "retrieves" in load_document(path)

def test_unsupported_extension(tmp_path):
    path = tmp_path / "file.docx"
    path.write_text("not supported", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(path)

def test_discovery_filters_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("ignored", encoding="utf-8")
    assert [p.name for p in discover_documents(tmp_path)] == ["a.txt"]
