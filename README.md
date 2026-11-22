# Milestone 2 — Document ingestion and parent-child chunking

Adds PDF/TXT loading, parent chunks (target 1,000 characters), child chunks
(target 200 characters), parent-child IDs, persistent JSON stores, and tests.

## Install
```powershell
python -m pip install -r requirements.txt
```

## Add documents
Place `.pdf` and `.txt` files in `data/documents/`. A sample TXT file is included.

## Run ingestion
```powershell
python -m app.ingestion.ingest
```

Generated local files:
- `data/parent_documents.json`
- `data/child_chunks.json`

## Test
```powershell
python -m pytest -v
```

Qdrant indexing and dense/sparse retrieval are deliberately left for M3.
