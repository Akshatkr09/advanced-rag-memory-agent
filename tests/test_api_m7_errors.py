from fastapi.testclient import TestClient

from app.api.routes import get_rag_service
from app.main import app


class FailingService:
    def chat(self, payload):
        raise RuntimeError("Controlled dependency failure.")

    def get_user_memories(self, user_id):
        raise RuntimeError("Controlled memory failure.")


def test_chat_dependency_failure_returns_503():
    app.dependency_overrides[get_rag_service] = lambda: FailingService()
    try:
        response = TestClient(app).post(
            "/api/chat",
            json={"query": "hello", "user_id": "u1", "session_id": "s1"},
        )
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
    assert response.status_code == 503
    assert response.json()["detail"] == "Controlled dependency failure."


def test_memory_dependency_failure_returns_503():
    app.dependency_overrides[get_rag_service] = lambda: FailingService()
    try:
        response = TestClient(app).get("/api/users/u1/memories")
    finally:
        app.dependency_overrides.pop(get_rag_service, None)
    assert response.status_code == 503
    assert response.json()["detail"] == "Controlled memory failure."
