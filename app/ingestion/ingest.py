import json
from pathlib import Path
from app.ingestion.chunking import create_parent_child_chunks, chunks_to_dicts
from app.ingestion.document_loader import discover_documents, load_document
from app.ingestion.parent_store import ParentDocumentStore

def ingest_directory(input_dir="data/documents",
                     parent_store_path="data/parent_documents.json",
                     child_output_path="data/child_chunks.json"):
    files = discover_documents(input_dir)
    if not files:
        return {"documents_processed": 0, "parents_created": 0, "children_created": 0,
                "message": f"No PDF or TXT documents found in {input_dir}"}
    store = ParentDocumentStore(parent_store_path)
    parents_all, children_all = [], []
    processed = 0
    for file_path in files:
        text = load_document(file_path)
        parents, children = create_parent_child_chunks(
            text, str(file_path), parent_size=1000, parent_overlap=100,
            child_size=200, child_overlap=30)
        if parents:
            processed += 1
            parents_all.extend(chunks_to_dicts(parents))
            children_all.extend(chunks_to_dicts(children))
    store.add_many(parents_all)
    child_path = Path(child_output_path)
    child_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if child_path.exists():
        existing = {x["child_id"]: x for x in json.loads(child_path.read_text(encoding="utf-8"))}
    for child in children_all:
        existing[child["child_id"]] = child
    temp = child_path.with_suffix(child_path.suffix + ".tmp")
    temp.write_text(json.dumps(list(existing.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(child_path)
    return {"documents_processed": processed, "parents_created": len(parents_all),
            "children_created": len(children_all), "parent_store": str(parent_store_path),
            "child_store": str(child_output_path)}

if __name__ == "__main__":
    print(json.dumps(ingest_directory(), indent=2))
