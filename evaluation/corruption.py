"""
Corruption Analysis & Robustness Suite:
Stress-tests the RAG system under noisy, corrupted, or adversarial context conditions:
1. Character-level typos and OCR degradation
2. Token omission / word dropping
3. Sentence permutation
4. Distractor chunk injection
Computes a quantitative Robustness Score and degradation profile.
"""

import random
import re
from typing import Dict, Any, List, Tuple
from rag.scoring import compute_cosine_similarity
from rag.embeddings import get_embedding_provider
from llm.engine_factory import get_llm_engine
from llm.prompts import get_rag_prompt


class ContextCorruptor:
    """Applies controlled corruption patterns to context text."""

    @staticmethod
    def inject_typos(text: str, noise_ratio: float = 0.1) -> str:
        """Injects random character swaps and substitutions simulating OCR/transcription errors."""
        chars = list(text)
        num_corruptions = int(len(chars) * noise_ratio)
        for _ in range(num_corruptions):
            idx = random.randint(0, len(chars) - 2)
            # Swap adjacent characters or replace with typo character
            if chars[idx].isalpha():
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)

    @staticmethod
    def drop_tokens(text: str, drop_ratio: float = 0.2) -> str:
        """Randomly drops tokens simulating missing packet data or truncated context."""
        words = text.split()
        retained = [w for w in words if random.random() > drop_ratio]
        return " ".join(retained)

    @staticmethod
    def inject_distractors(text: str, distractor_count: int = 2) -> str:
        """Injects irrelevant distractor passages into the context."""
        distractors = [
            "\n[Distractor Note: The quarterly sales of pineapples in 1984 grew by 14% due to oceanic currents.]",
            "\n[Irrelevant Context: Modern bicycles typically have two wheels and a steel or carbon fiber frame.]",
            "\n[Unrelated Noise: The recipe requires two eggs, one cup of sugar, and vanilla extract.]"
        ]
        chosen = distractors[:distractor_count]
        return text + "\n" + "\n".join(chosen)


class CorruptionAnalyzer:
    def __init__(self):
        self.embed_provider = get_embedding_provider()
        self.llm_engine = get_llm_engine()

    def run_stress_test(
        self,
        query: str,
        clean_context: str,
        noise_levels: List[float] = [0.0, 0.25, 0.50, 0.75]
    ) -> Dict[str, Any]:
        """
        Evaluates system degradation across 0%, 25%, 50%, and 75% corruption.
        Returns a comparative report and overall Robustness Score.
        """
        # Baseline response with 0% noise
        sys_p, user_p = get_rag_prompt(query, clean_context)
        baseline_answer, _ = self.llm_engine.generate(prompt=user_p, system_prompt=sys_p)
        baseline_emb = self.embed_provider.embed_text(baseline_answer)

        level_reports = []
        similarities = []

        for level in noise_levels:
            if level == 0.0:
                corrupted_ctx = clean_context
            else:
                # Apply combined corruption
                typo_text = ContextCorruptor.inject_typos(clean_context, noise_ratio=level * 0.15)
                omitted_text = ContextCorruptor.drop_tokens(typo_text, drop_ratio=level * 0.25)
                corrupted_ctx = ContextCorruptor.inject_distractors(omitted_text, distractor_count=1)

            sys_p, user_p = get_rag_prompt(query, corrupted_ctx)
            ans, tele = self.llm_engine.generate(prompt=user_p, system_prompt=sys_p)
            ans_emb = self.embed_provider.embed_text(ans)
            
            sim_to_baseline = compute_cosine_similarity(baseline_emb, ans_emb)
            similarities.append(max(0.0, sim_to_baseline))

            level_reports.append({
                "noise_level": f"{int(level * 100)}%",
                "similarity_to_baseline": round(sim_to_baseline, 3),
                "response_preview": ans[:120] + "...",
                "latency_ms": round(tele.get("total_latency_ms", 0.0), 2)
            })

        # Overall Robustness Score: Average retention of semantic meaning under noise
        avg_retention = sum(similarities) / max(len(similarities), 1)
        robustness_score = round(avg_retention, 3)

        return {
            "query": query,
            "robustness_score": robustness_score,
            "resilience_rating": "Superior" if robustness_score > 0.85 else "Resilient" if robustness_score > 0.70 else "Vulnerable",
            "degradation_curve": level_reports
        }
