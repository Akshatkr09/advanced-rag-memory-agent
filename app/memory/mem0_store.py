"""Mem0 OSS adapter configured to use Gemini and persistent local Qdrant."""
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def build_mem0_config() -> dict:
    """Build a Windows-friendly, persistent Mem0 OSS configuration."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your Gemini API key to the project's .env file."
        )

    qdrant_path = os.getenv("MEM0_QDRANT_PATH", "data/mem0_qdrant")
    history_path = os.getenv("MEM0_HISTORY_DB_PATH", "data/mem0_history.db")
    Path(qdrant_path).parent.mkdir(parents=True, exist_ok=True)
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)

    embedding_dims = int(os.getenv("MEM0_EMBEDDING_DIMS", "768"))
    return {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": os.getenv("MEM0_LLM_MODEL", "gemini-2.0-flash"),
                "api_key": api_key,
                "temperature": 0.1,
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": os.getenv(
                    "MEM0_EMBEDDING_MODEL", "models/gemini-embedding-001"
                ),
                "api_key": api_key,
                "embedding_dims": embedding_dims,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": os.getenv("MEM0_COLLECTION", "personal_memories"),
                "embedding_model_dims": embedding_dims,
                "path": qdrant_path,
                "on_disk": True,
            },
        },
        "history_db_path": history_path,
        "custom_instructions": (
            "Extract stable, useful user facts and preferences from conversations. "
            "Prioritize preferred programming languages, response style, tools, "
            "ongoing goals, and explicit preferences. Do not infer sensitive traits. "
            "Do not store temporary details unless they are clearly useful later."
        ),
    }


class Mem0Store:
    """Small injectable wrapper around Mem0's local Memory implementation."""

    def __init__(self, memory_client: Any | None = None):
        self._memory = memory_client

    def _client(self):
        if self._memory is None:
            try:
                from mem0 import Memory
            except ImportError as exc:
                raise RuntimeError(
                    "Mem0 is not installed. Run `python -m pip install -r requirements.txt`."
                ) from exc
            self._memory = Memory.from_config(build_mem0_config())
        return self._memory

    def add_conversation(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        session_id: str | None = None,
    ) -> dict:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty.")
        if not messages:
            raise ValueError("messages must contain at least one message.")
        return self._client().add(
            messages,
            user_id=user_id,
            run_id=session_id,
            infer=True,
        )

    def search_memories(self, query: str, user_id: str, limit: int = 5) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty.")
        response = self._client().search(
            query=query,
            user_id=user_id,
            limit=limit,
        )
        return response.get("results", []) if isinstance(response, dict) else response

    def get_user_memories(self, user_id: str, limit: int = 100) -> list[dict]:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty.")
        response = self._client().get_all(
            user_id=user_id,
            limit=limit,
        )
        return response.get("results", []) if isinstance(response, dict) else response

    def build_user_context(self, query: str, user_id: str, limit: int = 5) -> str:
        memories = self.search_memories(query, user_id, limit=limit)
        facts = [
            str(item.get("memory", "")).strip()
            for item in memories
            if item.get("memory")
        ]
        if not facts:
            return ""
        return "Relevant user preferences and facts:\n" + "\n".join(
            f"- {fact}" for fact in facts
        )
