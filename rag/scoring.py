"""
Advanced Scoring and Fusion Techniques:
1. Cosine Similarity & Vector Distance
2. BM25 Lexical Scoring with Token Normalization
3. Reciprocal Rank Fusion (RRF)
4. Alpha-Weighted Hybrid Scoring
5. Cross-Encoder / Contextual Re-ranking
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity between two vector representations."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-Max normalizes a list of arbitrary scores into the range [0.0, 1.0]."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


class BM25Scorer:
    """Performs sparse keyword BM25 retrieval across document chunks."""
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]
        if self.tokenized_corpus and any(len(t) > 0 for t in self.tokenized_corpus):
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and clean non-alphanumeric
        import re
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def score(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Returns list of (doc_index, raw_score) sorted descending by relevance."""
        if not self.bm25 or not self.corpus:
            return []
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []
        
        doc_scores = self.bm25.get_scores(tokenized_query)
        indexed_scores = [(idx, float(score)) for idx, score in enumerate(doc_scores) if score > 0]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF):
    Computes an integrated rank score across dense and sparse ranking lists:
    RRF_score(d) = sum_{system} 1 / (k + rank(d))
    """
    rrf_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict[str, Any]] = {}

    # Process dense results
    for rank, item in enumerate(dense_results, start=1):
        doc_key = item.get("text", "")
        if not doc_key:
            continue
        rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (k + rank))
        if doc_key not in doc_lookup:
            doc_lookup[doc_key] = {**item, "dense_rank": rank}
        else:
            doc_lookup[doc_key]["dense_rank"] = rank

    # Process sparse results
    for rank, item in enumerate(sparse_results, start=1):
        doc_key = item.get("text", "")
        if not doc_key:
            continue
        rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (k + rank))
        if doc_key not in doc_lookup:
            doc_lookup[doc_key] = {**item, "sparse_rank": rank}
        else:
            doc_lookup[doc_key]["sparse_rank"] = rank

    # Sort documents by accumulated RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results = []
    for doc_key, score in sorted_docs:
        entry = doc_lookup[doc_key]
        entry["fusion_score"] = float(score)
        entry["rrf_score"] = float(score)
        fused_results.append(entry)

    return fused_results


def weighted_hybrid_score(
    dense_score: float,
    sparse_score: float,
    alpha: float = 0.7
) -> float:
    """
    Linear combination of normalized dense and sparse scores:
    FinalScore = alpha * DenseScore + (1 - alpha) * SparseScore
    """
    alpha = max(0.0, min(1.0, alpha))
    return (alpha * dense_score) + ((1.0 - alpha) * sparse_score)


class FastReRanker:
    """
    Fast term-overlap and semantic cross-scoring re-ranker.
    Re-scores candidate chunks against the target query to prioritize precise matches.
    """
    @staticmethod
    def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        query_words = set(query.lower().split())
        reranked = []

        for item in candidates:
            text = item.get("text", "").lower()
            text_words = set(text.split())

            # Jaccard overlap
            intersection = query_words.intersection(text_words)
            jaccard = len(intersection) / max(len(query_words.union(text_words)), 1)

            # Exact phrase bonus
            phrase_bonus = 0.3 if query.lower() in text else 0.0

            # Base score from previous stage
            base_score = item.get("score") or item.get("fusion_score", 0.5)

            new_score = (0.5 * base_score) + (0.3 * jaccard) + (0.2 * phrase_bonus)
            item_copy = dict(item)
            item_copy["rerank_score"] = float(new_score)
            reranked.append(item_copy)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
