"""
Query Rewriter and Expansion Engine:
Rewrites ambiguous questions and generates multi-query variants for increased recall.
"""

from typing import List
from llm.engine_factory import get_llm_engine
from llm.prompts import QUERY_EXPANSION_PROMPT


class QueryRewriter:
    def __init__(self):
        self.engine = get_llm_engine()

    def expand_query(self, query: str) -> List[str]:
        """
        Generates query variations using LLM expansion with rule-based fallbacks.
        """
        prompt = QUERY_EXPANSION_PROMPT.format(query=query)
        try:
            expanded_text, _ = self.engine.generate(
                prompt=prompt,
                system_prompt="You are a query expansion specialist.",
                max_tokens=256,
                temperature=0.3
            )
            lines = [line.strip().lstrip("-*123456789. ") for line in expanded_text.split("\n") if line.strip()]
            queries = [query] + [q for q in lines if q and q.lower() != query.lower()]
            return queries[:4]
        except Exception:
            # Fallback heuristic expansion
            return [query, f"overview and details of {query}", f"{query} technical architecture"]

    def decompose_query(self, query: str) -> List[str]:
        """Splits multi-part questions into individual atomic questions."""
        if " and " in query.lower() or " vs " in query.lower() or " compared to " in query.lower():
            import re
            parts = re.split(r'\band\b|\bvs\b|\bcompared to\b', query, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts if len(p.strip()) > 3]
            if len(sub_queries) > 1:
                return sub_queries
        return [query]
