"""
RAG Package: Core retrieval-augmented generation modules.
"""
from .chunking import (
    RecursiveCharacterChunker,
    SemanticSentenceChunker,
    StructuredDocumentChunker,
    get_chunker
)
from .embeddings import (
    GeminiEmbeddingProvider,
    LocalFallbackEmbeddingProvider,
    get_embedding_provider
)
from .vector_store import QdrantVectorStore
from .scoring import (
    reciprocal_rank_fusion,
    normalize_scores,
    compute_cosine_similarity
)
from .simple_rag import SimpleRAG
from .hybrid_rag import HybridRAG
from .graph_rag import GraphRAG

__all__ = [
    "RecursiveCharacterChunker",
    "SemanticSentenceChunker",
    "StructuredDocumentChunker",
    "get_chunker",
    "GeminiEmbeddingProvider",
    "LocalFallbackEmbeddingProvider",
    "get_embedding_provider",
    "QdrantVectorStore",
    "reciprocal_rank_fusion",
    "normalize_scores",
    "compute_cosine_similarity",
    "SimpleRAG",
    "HybridRAG",
    "GraphRAG"
]
