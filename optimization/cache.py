"""
Semantic Response Cache:
Caches query embeddings and responses.
If a new query has cosine similarity >= cache_threshold (default 0.92) with a previous query,
it returns the cached response instantly without invoking the vector store or LLM.
"""

import time
from typing import Dict, Any, Optional, List, Tuple
from rag.scoring import compute_cosine_similarity
from rag.embeddings import get_embedding_provider


class SemanticCache:
    def __init__(self, threshold: float = 0.92, max_entries: int = 1000):
        self.threshold = threshold
        self.max_entries = max_entries
        self.cache: List[Dict[str, Any]] = []
        self.embed_provider = get_embedding_provider()

    def get(
        self,
        query: str,
        rag_type: Optional[str] = None,
        query_embedding: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Looks up the semantic cache with optional rag_type matching.
        """
        if not self.cache:
            return None

        start_time = time.perf_counter()
        q_emb = query_embedding or self.embed_provider.embed_text(query)

        best_score = 0.0
        best_entry = None

        for entry in self.cache:
            if rag_type and entry.get("rag_type") != rag_type:
                continue
            sim = compute_cosine_similarity(q_emb, entry["embedding"])
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            lookup_ms = (time.perf_counter() - start_time) * 1000
            return {
                "response": best_entry["response"],
                "similarity": round(best_score, 4),
                "matched_query": best_entry["query"],
                "lookup_latency_ms": round(lookup_ms, 2),
                "rag_type": best_entry.get("rag_type", "cached"),
                "context": best_entry.get("context", []),
                "extra": best_entry.get("extra", {})
            }

        return None

    def set(
        self,
        query: str,
        response: str,
        query_embedding: Optional[List[float]] = None,
        rag_type: str = "hybrid",
        context: Optional[List[Dict[str, Any]]] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Stores a new query-response pair in the semantic cache."""
        q_emb = query_embedding or self.embed_provider.embed_text(query)

        if len(self.cache) >= self.max_entries:
            self.cache.pop(0)

        self.cache.append({
            "query": query,
            "embedding": q_emb,
            "response": response,
            "rag_type": rag_type,
            "context": context or [],
            "extra": extra or {},
            "timestamp": time.time()
        })

    def clear(self):
        """Clears all cached entries."""
        self.cache.clear()

    def size(self) -> int:
        return len(self.cache)


# Global singleton instance
global_semantic_cache = SemanticCache()
