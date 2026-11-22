from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_chat_endpoint_accepts_required_fields():
    response = client.post(
        "/api/chat",
        json={"query": "hello", "user_id": "usr_test", "session_id": "s1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "usr_test"
    assert set(body["telemetry"]) == {
        "retrieval_latency_ms",
        "rerank_latency_ms",
        "generation_latency_ms",
        "total_latency_ms",
    }


def test_memories_endpoint_contract():
    response = client.get("/api/users/usr_test/memories")
    assert response.status_code == 200
    assert response.json() == {"user_id": "usr_test", "memories": []}
