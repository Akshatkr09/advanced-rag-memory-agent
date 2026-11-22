# Advanced RAG Agent with Context Compression & Semantic Memory

A Python project implementing the assessment brief's advanced RAG workflow:
hybrid dense/sparse retrieval, parent-document retrieval, Cohere reranking,
persistent semantic memory, FastAPI endpoints, and latency telemetry.

## Current milestone

Milestone 1: project scaffold and API skeleton. The chat and memory routes are
contract placeholders; the actual RAG and memory integrations will be added in
subsequent milestones. Do not treat placeholder telemetry as real measurements.

## Requirements

- Python 3.11 recommended
- Git
- API credentials will be configured in later milestones

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add credentials when the integrations are introduced.

## Run

```powershell
uvicorn app.main:app --reload
```

Open:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

## Test

```powershell
pytest -v
```

## Planned architecture

- `app/ingestion/`: document loading and parent/child chunking
- `app/retrieval/`: Qdrant, BM25, RRF, parent retrieval, Cohere reranking
- `app/memory/`: Mem0 persistent semantic memory
- `app/api/`: FastAPI routes
- `app/services/`: orchestration and telemetry
- `tests/`: automated tests

## Security

Never commit `.env`, API keys, or local database files. `.env.example` contains
variable names only.
