"""Shared Gemini embedding configuration for indexing and querying."""
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBEDDING_MODEL = "gemini-embedding-001"

def get_embeddings():
    """Return the same Gemini embedding model for documents and queries."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add your Gemini API key to the project's .env file."
        )
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )
