"""
Embedding Layer:
1. GeminiEmbeddingProvider (Google GenAI text-embedding-004)
2. LocalFallbackEmbeddingProvider (Fast deterministic 768-dim semantic hashing for offline/testing)
"""

import os
import hashlib
import numpy as np
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


class BaseEmbeddingProvider:
    dimension: int = 768

    def embed_text(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Produces high-fidelity semantic embeddings using Google Gemini's text-embedding-004.
    Automatically falls back to LocalFallbackEmbeddingProvider if GEMINI_API_KEY is not set or fails.
    """
    def __init__(self, model: str = "text-embedding-004", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.dimension = 768
        self._client = None
        self._fallback = LocalFallbackEmbeddingProvider(dimension=self.dimension)

        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except Exception as e:
                print(f"[GeminiEmbeddingProvider] Warning: Could not configure google.generativeai: {e}")

    def _get_model_str(self) -> str:
        return self.model if self.model.startswith("models/") else f"models/{self.model}"

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension

        if self._client:
            model_target = self._get_model_str()
            for m in [model_target, "models/gemini-embedding-001"]:
                try:
                    result = self._client.embed_content(
                        model=m,
                        content=text,
                        task_type="retrieval_document",
                        output_dimensionality=self.dimension
                    )
                    embedding = result["embedding"]
                    return embedding[:self.dimension]
                except TypeError:
                    try:
                        result = self._client.embed_content(
                            model=m,
                            content=text,
                            task_type="retrieval_document"
                        )
                        return result["embedding"][:self.dimension]
                    except Exception:
                        continue
                except Exception:
                    continue

        return self._fallback.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if self._client:
            model_target = self._get_model_str()
            for m in [model_target, "models/gemini-embedding-001"]:
                try:
                    results = self._client.embed_content(
                        model=m,
                        content=texts,
                        task_type="retrieval_document",
                        output_dimensionality=self.dimension
                    )
                    embeddings = results.get("embedding", [])
                    if embeddings and isinstance(embeddings[0], list):
                        return [emb[:self.dimension] for emb in embeddings]
                except Exception:
                    continue
        
        # Fallback or loop
        return [self.embed_text(t) for t in texts]


class LocalFallbackEmbeddingProvider(BaseEmbeddingProvider):
    """
    Deterministic 768-dimensional normalized embedding using token n-gram hashing
    and character trigrams. Guarantees consistent vector space for offline and local execution.
    """
    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self.dimension

        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = text.lower().split()

        for token in tokens:
            # Word-level hash projection
            h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign * 1.5

            # Sub-token trigram projection
            for i in range(len(token) - 2):
                trigram = token[i:i+3]
                th = int(hashlib.sha256(trigram.encode('utf-8')).hexdigest(), 16)
                tidx = th % self.dimension
                tsign = 1.0 if (th >> 12) % 2 == 0 else -1.0
                vec[tidx] += tsign * 0.8

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def get_embedding_provider(strategy: str = "gemini") -> BaseEmbeddingProvider:
    """Returns the requested embedding provider."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if strategy.lower() == "gemini" and api_key and api_key != "your_gemini_api_key_here":
        return GeminiEmbeddingProvider(api_key=api_key)
    return LocalFallbackEmbeddingProvider(dimension=768)
