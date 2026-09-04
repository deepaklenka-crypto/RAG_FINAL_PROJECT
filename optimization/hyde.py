"""
Hypothetical Document Embeddings (HyDE):
Generates a hypothetical document that directly answers the question,
then embeds that hypothetical document for semantic vector search in Qdrant.
"""

from typing import List, Optional
from llm.engine_factory import get_llm_engine
from llm.prompts import HYDE_PROMPT
from rag.embeddings import get_embedding_provider


class HyDEGenerator:
    def __init__(self):
        self.engine = get_llm_engine()
        self.embed_provider = get_embedding_provider()

    def generate_hypothetical_embedding(self, query: str) -> List[float]:
        """
        Generates a hypothetical answer passage and returns its vector embedding.
        """
        prompt = HYDE_PROMPT.format(query=query)
        try:
            hypothetical_doc, _ = self.engine.generate(
                prompt=prompt,
                system_prompt="You are a domain expert drafting a concise answer passage.",
                max_tokens=256,
                temperature=0.5
            )
        except Exception:
            hypothetical_doc = query

        # Embed the hypothetical document
        return self.embed_provider.embed_text(hypothetical_doc)
