"""Qdrant-backed storage for child chunks and their dense vectors."""
import os
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "child_chunks")

def _point_id(child_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, child_id))

class QdrantChildStore:
    def __init__(self, client=None, collection_name=None, local_path=None):
        self.collection_name = collection_name or COLLECTION_NAME
        if client is not None:
            self.client = client
        else:
            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY") or None
            if url:
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                path = local_path or os.getenv("QDRANT_LOCAL_PATH", "data/qdrant")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=path)

    def ensure_collection(self, vector_size: int):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
            )
        else:
            info = self.client.get_collection(self.collection_name)
            existing_size = info.config.params.vectors.size
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' has vector size "
                    f"{existing_size}; expected {vector_size}. Use a fresh collection "
                    "or remove the old local Qdrant data before changing embedding models."
                )

    def upsert(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError("Each child chunk must have exactly one embedding.")
        if not chunks:
            return 0
        self.ensure_collection(len(vectors[0]))
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(models.PointStruct(
                id=_point_id(chunk["child_id"]),
                vector=vector,
                payload={
                    "child_id": chunk["child_id"],
                    "parent_id": chunk["parent_id"],
                    "document_id": chunk["document_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                },
            ))
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    def search(self, query_vector, limit=20):
        if not self.client.collection_exists(self.collection_name):
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        results = []
        for point in response.points:
            item = dict(point.payload or {})
            item["dense_score"] = float(point.score)
            results.append(item)
        return results
