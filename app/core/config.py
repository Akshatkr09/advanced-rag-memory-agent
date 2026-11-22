from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Advanced RAG Memory Agent"
    openai_api_key: str = ""
    cohere_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    mem0_api_key: str = ""
    data_dir: str = "data"
    rerank_model: str = "rerank-english-v3.0"
    rerank_top_n: int = 5
    retrieval_top_k: int = 20
    rerank_min_score: float = 0.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
