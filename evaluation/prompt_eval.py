"""
Prompt-Level Evaluation:
Measures prompt clarity, token efficiency, injection resistance, and ambiguity.
"""

import re
from typing import Dict, Any


class PromptEvaluator:
    # Known prompt injection signatures
    INJECTION_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"system prompt override",
        r"reveal your system prompt",
        r"disregard (all )?prior rules",
        r"act as a jailbroken",
        r"dan mode",
        r"developer mode enabled"
    ]

    @classmethod
    def evaluate(cls, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        """
        Calculates prompt-level quality metrics:
        1. injection_risk_score (0.0 to 1.0)
        2. clarity_score (0.0 to 1.0)
        3. token_efficiency (0.0 to 1.0)
        4. ambiguity_score (0.0 to 1.0)
        """
        clean_p = prompt.strip().lower()
        word_count = len(clean_p.split())

        # 1. Injection Risk Check
        injection_hits = 0
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, clean_p):
                injection_hits += 1
        injection_risk = min(1.0, injection_hits * 0.5)

        # 2. Ambiguity Score (very short or vague queries)
        vague_terms = ["tell me", "what about it", "explain this", "help", "info", "data"]
        ambiguity = 0.0
        if word_count < 3:
            ambiguity += 0.5
        if any(clean_p == v for v in vague_terms):
            ambiguity += 0.4
        ambiguity = min(1.0, ambiguity)

        # 3. Clarity Score
        clarity = 1.0 - (ambiguity * 0.7 + injection_risk * 0.3)
        clarity = max(0.1, round(clarity, 2))

        # 4. Token Efficiency (penalizes excessive repetitive bloat)
        unique_words = len(set(clean_p.split()))
        repetition_ratio = unique_words / max(word_count, 1)
        token_efficiency = round(min(1.0, repetition_ratio * 1.1), 2)

        return {
            "prompt_length_words": word_count,
            "clarity_score": clarity,
            "token_efficiency": token_efficiency,
            "injection_risk_score": round(injection_risk, 2),
            "ambiguity_score": round(ambiguity, 2),
            "passed_safety": (injection_risk < 0.5)
        }
