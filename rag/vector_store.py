"""
Qdrant Vector Database Integration:
Supports local on-disk persistence (default) and remote Qdrant cluster connections.
Manages collections, point upserts, cosine similarity search, and payload filtering.
"""

import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()

# Global cache for QdrantClient instances to prevent file lock conflicts in local disk mode
_GLOBAL_QDRANT_CLIENTS: Dict[str, QdrantClient] = {}


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: Optional[str] = None,
        dimension: int = 768,
        storage_path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "rag_multi_collection")
        self.dimension = dimension
        self.url = url or os.getenv("QDRANT_URL", "").strip()
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", "").strip()
        self.storage_path = os.path.abspath(storage_path or os.getenv("QDRANT_PATH", "./data/qdrant_storage"))

        cache_key = self.url if self.url else self.storage_path
        if cache_key in _GLOBAL_QDRANT_CLIENTS:
            self.client = _GLOBAL_QDRANT_CLIENTS[cache_key]
        else:
            if self.url:
                self.client = QdrantClient(url=self.url, api_key=self.api_key or None)
            else:
                os.makedirs(self.storage_path, exist_ok=True)
                try:
                    self.client = QdrantClient(path=self.storage_path)
                except Exception as e:
                    # In case another process (e.g. active main.py server) is locking the on-disk storage folder
                    self.client = QdrantClient(":memory:")
            _GLOBAL_QDRANT_CLIENTS[cache_key] = self.client

        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the Qdrant collection if it does not already exist."""
        try:
            collections = self.client.get_collections().collections
            existing = [c.name for c in collections]
            if self.collection_name not in existing:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
                )
        except Exception as e:
            # Fallback for local storage lock or existing check
            pass

    def upsert_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Upserts text chunks and their embeddings with metadata payloads."""
        if not chunks or not embeddings:
            return []

        points = []
        assigned_ids = []

        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            pt_id = ids[i] if (ids and i < len(ids)) else str(uuid.uuid4())
            assigned_ids.append(pt_id)
            meta = metadatas[i] if (metadatas and i < len(metadatas)) else {}
            payload = {
                "text": text,
                **meta
            }
            points.append(
                PointStruct(
                    id=pt_id,
                    vector=emb,
                    payload=payload
                )
            )

        # Batch upsert
        batch_size = 100
        for b in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[b : b + batch_size]
            )

        return assigned_ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_doc_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes dense cosine similarity search in Qdrant.
        Returns a list of dictionaries with text, score, and payload metadata.
        """
        query_filter = None
        if filter_doc_id is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=filter_doc_id)
                    )
                ]
            )

        try:
            # query_points handles standard Qdrant queries
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter
            ).points
        except Exception:
            # Backward compatibility with older search() method
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

        results = []
        for hit in search_result:
            payload = hit.payload or {}
            results.append({
                "id": str(hit.id),
                "score": float(hit.score),
                "text": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"}
            })

        return results

    def delete_document_chunks(self, document_id: int):
        """Deletes all chunks belonging to a specific document ID."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            print(f"[QdrantVectorStore] Warning during deletion: {e}")

    def count(self) -> int:
        """Returns the total number of points in the collection."""
        try:
            res = self.client.count(collection_name=self.collection_name)
            return res.count
        except Exception:
            return 0
