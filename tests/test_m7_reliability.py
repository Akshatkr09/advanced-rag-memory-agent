import pytest
from langchain_core.documents import Document

from app.services.rag_service import RAGService, RAGServiceError


class MemoryByUser:
    def __init__(self):
        self.facts = {
            "user_a": ["Prefers Python"],
            "user_b": ["Prefers Java"],
        }
        self.saved = []

    def build_user_context(self, query, user_id):
        return "\n".join(self.facts.get(user_id, []))

    def add_conversation(self, messages, user_id, session_id):
        self.saved.append((user_id, session_id, messages))
        return {}

    def get_user_memories(self, user_id):
        return [{"memory": fact} for fact in self.facts.get(user_id, [])]


class EmptyRetriever:
    def search(self, query, limit=20):
        return {"matches": [], "parents": []}


class BrokenRetriever:
    def search(self, query, limit=20):
        raise ConnectionError("private infrastructure detail")


class BrokenMemory(MemoryByUser):
    def build_user_context(self, query, user_id):
        raise ConnectionError("private memory detail")

    def get_user_memories(self, user_id):
        raise ConnectionError("private memory detail")


class FakeCompressor:
    def compress_documents(self, docs, query):
        return [
            Document(
                page_content=d.page_content,
                metadata={**d.metadata, "relevance_score": 0.9},
            )
            for d in docs
        ]


def make_service(memory=None, retriever=None, generator=None):
    return RAGService(
        hybrid_retriever=retriever or EmptyRetriever(),
        memory_store=memory or MemoryByUser(),
        compressor=FakeCompressor(),
        generation_callable=generator or (lambda **kwargs: "A concise answer."),
    )


def payload(user_id="user_a", session_id="s1", query="Explain Python"):
    return {"query": query, "user_id": user_id, "session_id": session_id}


def test_memory_context_is_scoped_to_requested_user():
    captured = []
    service = make_service(generator=lambda **kwargs: captured.append(kwargs["memory_context"]) or "ok")
    service.chat(payload("user_a"))
    service.chat(payload("user_b", "s2"))
    assert "Prefers Python" in captured[0]
    assert "Prefers Java" not in captured[0]
    assert "Prefers Java" in captured[1]
    assert "Prefers Python" not in captured[1]


def test_retrieval_dependency_failure_becomes_controlled_error():
    service = make_service(retriever=BrokenRetriever())
    with pytest.raises(RAGServiceError, match="Document retrieval failed"):
        service.chat(payload())


def test_memory_lookup_failure_becomes_controlled_error():
    service = make_service(memory=BrokenMemory())
    with pytest.raises(RAGServiceError, match="User memory lookup failed"):
        service.chat(payload())


def test_invalid_blank_fields_are_rejected():
    service = make_service()
    with pytest.raises(ValueError, match="query"):
        service.chat(payload(query="   "))
    with pytest.raises(ValueError, match="user_id"):
        service.chat(payload(user_id=" "))
    with pytest.raises(ValueError, match="session_id"):
        service.chat(payload(session_id=""))


def test_empty_answer_is_rejected_and_not_saved():
    memory = MemoryByUser()
    service = make_service(memory=memory, generator=lambda **kwargs: " ")
    with pytest.raises(RAGServiceError, match="empty or invalid answer"):
        service.chat(payload())
    assert memory.saved == []


def test_memory_save_failure_is_reported():
    class SaveFailureMemory(MemoryByUser):
        def add_conversation(self, messages, user_id, session_id):
            raise OSError("private database detail")
    service = make_service(memory=SaveFailureMemory())
    with pytest.raises(RAGServiceError, match="could not be saved"):
        service.chat(payload())


def test_empty_index_still_returns_valid_positive_telemetry():
    service = make_service()
    result = service.chat(payload())
    assert result["sources"] == []
    assert result["answer"] == "A concise answer."
    assert all(isinstance(v, (int, float)) and v > 0 for v in result["telemetry"].values())


def test_memory_endpoint_failure_is_controlled():
    service = make_service(memory=BrokenMemory())
    with pytest.raises(RAGServiceError, match="User memory retrieval failed"):
        service.get_user_memories("user_a")
