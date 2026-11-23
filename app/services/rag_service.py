"""End-to-end RAG, semantic-memory, and Gemini orchestration."""
import os
import time
from typing import Any, Callable
from dotenv import load_dotenv
from app.ingestion.parent_store import ParentDocumentStore
from app.memory.mem0_store import Mem0Store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import create_cohere_reranker, rerank_candidates

SYSTEM_PROMPT = """You are an assistant that answers the user's question using the
provided knowledge context when it is relevant.

Use retrieved document context as supporting information. If the context does not
contain the answer, be transparent and use your general knowledge carefully.

The user profile below contains preferences and facts explicitly remembered from
earlier conversations. Apply relevant preferences naturally, especially preferred
programming language and response style. Do not repeat the profile unless useful.

{memory_context}

Knowledge context:
{knowledge_context}
"""

def _positive_ms(start: float, end: float) -> float:
    return max((end - start) * 1000.0, 0.001)

class RAGService:
    """Coordinates hybrid retrieval, Cohere reranking, Mem0, and Gemini."""
    def __init__(
        self,
        hybrid_retriever: Any | None = None,
        memory_store: Any | None = None,
        compressor: Any | None = None,
        parent_store: Any | None = None,
        generation_callable: Callable[..., str] | None = None,
        generation_model: str | None = None,
    ):
        self._hybrid_retriever = hybrid_retriever
        self.memory_store = memory_store or Mem0Store()
        self._compressor = compressor
        self.parent_store = parent_store
        self.generation_callable = generation_callable
        self.generation_model = generation_model

    def _get_hybrid_retriever(self):
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever()
        return self._hybrid_retriever

    def _get_compressor(self):
        if self._compressor is None:
            load_dotenv()
            self._compressor = create_cohere_reranker(
                top_n=int(os.getenv("RERANK_TOP_N", "5"))
            )
        return self._compressor

    def _get_parent_store(self):
        if self.parent_store is None:
            self.parent_store = ParentDocumentStore(
                os.getenv("PARENT_STORE_PATH", "data/parent_documents.json")
            )
        return self.parent_store

    def _generate(self, query: str, memory_context: str, knowledge_context: str) -> str:
        system_prompt = SYSTEM_PROMPT.format(
            memory_context=memory_context or "(No saved user preferences found.)",
            knowledge_context=knowledge_context or "(No matching documents were retrieved.)",
        )
        if self.generation_callable is not None:
            return self.generation_callable(
                query=query, system_prompt=system_prompt,
                memory_context=memory_context, knowledge_context=knowledge_context,
            )
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add your Gemini API key to the project's .env file.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Google GenAI SDK is missing. Run `python -m pip install -r requirements.txt`.") from exc
        client = genai.Client(api_key=api_key)
        model = self.generation_model or os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
        response = client.models.generate_content(
            model=model,
            contents=query,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.2),
        )
        answer = getattr(response, "text", None)
        if not answer:
            raise RuntimeError("Gemini returned an empty response.")
        return answer.strip()

    def chat(self, payload: Any) -> dict:
        query = payload.query if hasattr(payload, "query") else payload["query"]
        user_id = payload.user_id if hasattr(payload, "user_id") else payload["user_id"]
        session_id = payload.session_id if hasattr(payload, "session_id") else payload["session_id"]
        if not query or not query.strip():
            raise ValueError("query must not be empty.")
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty.")
        if not session_id or not session_id.strip():
            raise ValueError("session_id must not be empty.")

        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        try:
            retrieval_result = self._get_hybrid_retriever().search(query, limit=20)
        except FileNotFoundError:
            retrieval_result = {"matches": [], "parents": []}
        memory_context = self.memory_store.build_user_context(query, user_id)
        retrieval_latency_ms = _positive_ms(retrieval_start, time.perf_counter())

        rerank_start = time.perf_counter()
        candidates = retrieval_result.get("matches", [])
        selected = []
        if candidates:
            load_dotenv()
            selected = rerank_candidates(
                query=query,
                candidates=candidates,
                compressor=self._get_compressor(),
                top_n=int(os.getenv("RERANK_TOP_N", "5")),
                min_score=float(os.getenv("RERANK_MIN_SCORE", "0.0")),
            )
        parent_ids = [item["parent_id"] for item in selected if item.get("parent_id")]
        parent_docs = self._get_parent_store().get_many(parent_ids) if parent_ids else []
        if parent_docs:
            knowledge_context = "\n\n".join(
                str(item.get("text", "")).strip() for item in parent_docs if item.get("text")
            )
        else:
            knowledge_context = "\n\n".join(
                str(item.get("text", "")).strip() for item in selected if item.get("text")
            )
        rerank_latency_ms = _positive_ms(rerank_start, time.perf_counter())

        generation_start = time.perf_counter()
        answer = self._generate(query, memory_context, knowledge_context)
        generation_latency_ms = _positive_ms(generation_start, time.perf_counter())

        self.memory_store.add_conversation(
            [{"role": "user", "content": query}, {"role": "assistant", "content": answer}],
            user_id=user_id, session_id=session_id,
        )
        total_end = time.perf_counter()

        sources = []
        for source in ([x.get("source") for x in parent_docs] or [x.get("source") for x in selected]):
            if source and source not in sources:
                sources.append(source)
        return {
            "answer": answer,
            "user_id": user_id,
            "session_id": session_id,
            "sources": sources,
            "telemetry": {
                "retrieval_latency_ms": retrieval_latency_ms,
                "rerank_latency_ms": rerank_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "total_latency_ms": _positive_ms(total_start, total_end),
            },
        }

    def get_user_memories(self, user_id: str) -> list[dict]:
        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty.")
        return self.memory_store.get_user_memories(user_id)
