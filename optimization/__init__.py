"""
Optimization Package: Semantic Cache, HyDE, Query Expansion, and Context Compression.
"""
from .cache import SemanticCache, global_semantic_cache
from .query_rewriter import QueryRewriter
from .hyde import HyDEGenerator
from .compression import ContextCompressor

__all__ = [
    "SemanticCache",
    "global_semantic_cache",
    "QueryRewriter",
    "HyDEGenerator",
    "ContextCompressor"
]
