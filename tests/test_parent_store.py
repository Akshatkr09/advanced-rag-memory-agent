from app.ingestion.parent_store import ParentDocumentStore

def test_parent_store_persistence(tmp_path):
    path = tmp_path / "parents.json"
    store = ParentDocumentStore(path)
    parent = {"parent_id":"p1","document_id":"d1","source":"x.txt",
              "text":"parent content","start_char":0,"end_char":14}
    store.add_many([parent])
    loaded = ParentDocumentStore(path)
    assert loaded.get("p1") == parent
    assert loaded.get_many(["p1", "missing"]) == [parent]
