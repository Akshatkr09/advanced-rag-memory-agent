from fastapi.testclient import TestClient
from app.api.routes import get_rag_service
from app.main import app

class FakeAPIService:
    def chat(self, payload):
        return {
            "answer": "Hello from the test service.",
            "user_id": payload.user_id,
            "session_id": payload.session_id,
            "sources": [],
            "telemetry": {
                "retrieval_latency_ms": 1.0, "rerank_latency_ms": 1.0,
                "generation_latency_ms": 1.0, "total_latency_ms": 3.0,
            },
        }
    def get_user_memories(self, user_id):
        return [{"memory": "Works in Python"}]

client = TestClient(app)

def test_chat_endpoint_accepts_required_fields():
    app.dependency_overrides[get_rag_service] = lambda: FakeAPIService()
    try:
        response = client.post("/api/chat", json={
            "query": "hello", "user_id": "usr_test", "session_id": "s1"})
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "usr_test"
    assert body["session_id"] == "s1"
    assert body["answer"] == "Hello from the test service."
    assert set(body["telemetry"]) == {
        "retrieval_latency_ms", "rerank_latency_ms",
        "generation_latency_ms", "total_latency_ms"}

def test_memories_endpoint_returns_user_facts():
    app.dependency_overrides[get_rag_service] = lambda: FakeAPIService()
    try:
        response = client.get("/api/users/usr_test/memories")
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
    assert response.status_code == 200
    assert response.json() == {
        "user_id": "usr_test", "memories": [{"memory": "Works in Python"}]}

def test_chat_endpoint_validates_required_fields():
    response = client.post("/api/chat", json={"query": "hello"})
    assert response.status_code == 422
