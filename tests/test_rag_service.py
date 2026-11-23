from langchain_core.documents import Document
from app.services.rag_service import RAGService

class FakeHybridRetriever:
    def __init__(self):
        self.calls = []
    def search(self, query, limit=20):
        self.calls.append((query, limit))
        return {"matches": [{
            "child_id": "child-1", "parent_id": "parent-1",
            "source": "guide.txt", "text": "short child passage", "rrf_score": 0.03}],
            "parents": []}

class FakeCompressor:
    def compress_documents(self, documents, query):
        return [Document(page_content=doc.page_content,
            metadata={**doc.metadata, "relevance_score": 0.92}) for doc in documents]

class FakeMemoryStore:
    def __init__(self):
        self.added, self.context_queries = [], []
    def build_user_context(self, query, user_id):
        self.context_queries.append((query, user_id))
        return "Relevant user preferences and facts:\n- Works in Python\n- Prefers brief, code-only responses"
    def add_conversation(self, messages, user_id, session_id):
        self.added.append((messages, user_id, session_id))
        return {"results": []}
    def get_user_memories(self, user_id):
        return [{"memory": "Works in Python"}]

class FakeParentStore:
    def get_many(self, parent_ids):
        assert parent_ids == ["parent-1"]
        return [{"parent_id": "parent-1", "source": "guide.txt",
                 "text": "Full parent document context."}]

def test_chat_orchestrates_rag_memory_generation_and_telemetry():
    hybrid, memory, generated = FakeHybridRetriever(), FakeMemoryStore(), {}
    def fake_generate(**kwargs):
        generated.update(kwargs)
        return "print('merge sort')"
    service = RAGService(hybrid_retriever=hybrid, memory_store=memory,
        compressor=FakeCompressor(), parent_store=FakeParentStore(),
        generation_callable=fake_generate)
    result = service.chat({"query": "Write a quick merge sort routine.",
        "user_id": "usr_99", "session_id": "session_2"})
    assert result["answer"] == "print('merge sort')"
    assert result["sources"] == ["guide.txt"]
    assert "Works in Python" in generated["system_prompt"]
    assert "Full parent document context." in generated["knowledge_context"]
    assert hybrid.calls == [("Write a quick merge sort routine.", 20)]
    assert memory.context_queries == [("Write a quick merge sort routine.", "usr_99")]
    assert memory.added[0][1:] == ("usr_99", "session_2")
    assert [m["role"] for m in memory.added[0][0]] == ["user", "assistant"]
    for value in result["telemetry"].values():
        assert isinstance(value, (int, float)) and value > 0

def test_chat_works_without_an_indexed_knowledge_base():
    memory = FakeMemoryStore()
    class MissingIndex:
        def search(self, query, limit=20):
            raise FileNotFoundError("No child index yet")
    service = RAGService(hybrid_retriever=MissingIndex(), memory_store=memory,
        generation_callable=lambda **kwargs: "A general answer.")
    result = service.chat({"query": "What is Python?", "user_id": "usr_1",
        "session_id": "session_1"})
    assert result["answer"] == "A general answer."
    assert result["sources"] == []
    assert all(value > 0 for value in result["telemetry"].values())

def test_get_user_memories_delegates_to_mem0():
    service = RAGService(memory_store=FakeMemoryStore())
    assert service.get_user_memories("usr_99") == [{"memory": "Works in Python"}]
