import pytest

from app.memory.mem0_store import Mem0Store, build_mem0_config


class FakeMem0:
    def __init__(self):
        self.add_calls = []
        self.search_calls = []
        self.all_calls = []

    def add(self, messages, **kwargs):
        self.add_calls.append((messages, kwargs))
        return {"results": [{"id": "m1", "memory": "Works in Python"}]}

    def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return {"results": [
            {"id": "m1", "memory": "Works in Python", "score": 0.91},
            {"id": "m2", "memory": "Prefers brief, code-only responses", "score": 0.88},
        ]}

    def get_all(self, **kwargs):
        self.all_calls.append(kwargs)
        return {"results": [
            {"id": "m1", "memory": "Works in Python"},
            {"id": "m2", "memory": "Prefers brief, code-only responses"},
        ]}


def test_add_conversation_scopes_memory_to_user_and_session():
    fake = FakeMem0()
    store = Mem0Store(memory_client=fake)
    messages = [{"role": "user", "content": "I work in Python."}]

    result = store.add_conversation(messages, "usr_99", "session_1")

    assert result["results"][0]["memory"] == "Works in Python"
    assert fake.add_calls[0][1]["user_id"] == "usr_99"
    assert fake.add_calls[0][1]["run_id"] == "session_1"
    assert fake.add_calls[0][1]["infer"] is True


def test_search_uses_user_filter_and_returns_memories():
    fake = FakeMem0()
    store = Mem0Store(memory_client=fake)

    results = store.search_memories("merge sort coding preferences", "usr_99")

    assert len(results) == 2
    assert fake.search_calls[0][1]["user_id"] == "usr_99"


def test_get_user_memories_returns_all_user_facts():
    fake = FakeMem0()
    store = Mem0Store(memory_client=fake)

    results = store.get_user_memories("usr_99")

    assert len(results) == 2
    assert fake.all_calls[0]["user_id"] == "usr_99"


def test_build_user_context_formats_preferences_for_prompt():
    store = Mem0Store(memory_client=FakeMem0())

    context = store.build_user_context("Write merge sort", "usr_99")

    assert "Works in Python" in context
    assert "brief, code-only responses" in context


def test_rejects_empty_user_id():
    store = Mem0Store(memory_client=FakeMem0())
    with pytest.raises(ValueError):
        store.get_user_memories(" ")


def test_rejects_empty_conversation():
    store = Mem0Store(memory_client=FakeMem0())
    with pytest.raises(ValueError):
        store.add_conversation([], "usr_99")


def test_build_mem0_config_uses_gemini_and_local_qdrant(monkeypatch, tmp_path):
    monkeypatch.setattr("app.memory.mem0_store.load_dotenv", lambda: False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("MEM0_QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("MEM0_HISTORY_DB_PATH", str(tmp_path / "history.db"))

    config = build_mem0_config()

    assert config["llm"]["provider"] == "gemini"
    assert config["embedder"]["provider"] == "gemini"
    assert config["vector_store"]["provider"] == "qdrant"
    assert config["vector_store"]["config"]["on_disk"] is True
    assert config["vector_store"]["config"]["embedding_model_dims"] == 768


def test_mem0_config_requires_gemini_key(monkeypatch):
    # Prevent dotenv from loading the real project .env file in this test.
    monkeypatch.setattr("app.memory.mem0_store.load_dotenv", lambda: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        build_mem0_config()
