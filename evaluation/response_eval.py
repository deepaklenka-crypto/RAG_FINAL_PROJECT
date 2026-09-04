"""
Response-Level Evaluation:
Evaluates Faithfulness, Hallucination Rate, Answer Relevancy, and Format Adherence.
"""

import re
from typing import Dict, Any, List
from rag.scoring import compute_cosine_similarity
from rag.embeddings import get_embedding_provider


class ResponseEvaluator:
    def __init__(self):
        self.embed_provider = get_embedding_provider()

    def evaluate(
        self,
        query: str,
        response: str,
        context_passages: List[str],
        expected_mode: str = "general",
        expected_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Computes response-level metrics:
        - faithfulness_score (0.0 - 1.0)
        - hallucination_rate (0.0 - 1.0)
        - answer_relevancy_score (0.0 - 1.0)
        - format_adherence_score (0.0 - 1.0)
        """
        combined_context = " ".join(context_passages).lower()
        resp_clean = response.lower()

        # 1. Faithfulness Score & Hallucination Check
        # Extract response sentences / claims
        sentences = [s.strip() for s in re.split(r'[.!?]+', resp_clean) if len(s.strip()) > 10]
        grounded_count = 0

        for sent in sentences:
            sent_words = set(re.findall(r'\w+', sent))
            # If word overlap with context is high, sentence is grounded
            context_words = set(re.findall(r'\w+', combined_context))
            overlap = len(sent_words.intersection(context_words))
            if overlap / max(len(sent_words), 1) > 0.35:
                grounded_count += 1

        total_claims = max(len(sentences), 1)
        faithfulness = round(grounded_count / total_claims, 2)
        hallucination_rate = round(1.0 - faithfulness, 2)

        # 2. Answer Relevancy (semantic similarity between query and response)
        q_emb = self.embed_provider.embed_text(query)
        r_emb = self.embed_provider.embed_text(response[:500])
        semantic_sim = compute_cosine_similarity(q_emb, r_emb)
        answer_relevancy = round(max(0.0, min(1.0, (semantic_sim + 1.0) / 2.0)), 2)

        # 3. Format Adherence
        format_adherence = 1.0
        if expected_mode.lower() == "code":
            has_code_fence = "```python" in response or "```" in response or "def " in response
            format_adherence = 1.0 if has_code_fence else 0.4
        elif expected_language.lower() in ["hi", "hindi"]:
            # Check for Devanagari Unicode range \u0900-\u097F
            devanagari_chars = re.findall(r'[\u0900-\u097F]', response)
            format_adherence = 1.0 if len(devanagari_chars) > 20 else 0.5

        return {
            "faithfulness_score": faithfulness,
            "hallucination_rate": hallucination_rate,
            "answer_relevancy_score": answer_relevancy,
            "format_adherence_score": format_adherence,
            "total_claims_analyzed": len(sentences),
            "grounded_claims": grounded_count
        }
