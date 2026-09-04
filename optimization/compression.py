"""
Context Compression and Selective Pruning:
Prunes low-relevance sentences from retrieved chunks to reduce token overhead,
eliminate distractor noise, and improve LLM generation speed.
"""

import re
from typing import List, Dict, Any


class ContextCompressor:
    def __init__(self, relevance_threshold: float = 0.15, max_tokens: int = 1500):
        self.relevance_threshold = relevance_threshold
        self.max_tokens = max_tokens
        self.sentence_regex = re.compile(r'(?<=[.!?])\s+')

    def compress_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Extractively filters and orders the most relevant sentences across candidate chunks.
        """
        if not chunks:
            return ""

        query_tokens = set(re.findall(r'\w+', query.lower()))
        selected_sentences = []
        total_tokens = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = [s.strip() for s in self.sentence_regex.split(text) if s.strip()]
            
            for sent in sentences:
                sent_tokens = set(re.findall(r'\w+', sent.lower()))
                if not sent_tokens:
                    continue

                # Jaccard overlap score
                overlap = len(query_tokens.intersection(sent_tokens))
                rel_score = overlap / max(len(query_tokens), 1)

                if rel_score >= self.relevance_threshold or len(selected_sentences) < 2:
                    sent_token_len = len(sent.split())
                    if total_tokens + sent_token_len <= self.max_tokens:
                        selected_sentences.append((sent, rel_score))
                        total_tokens += sent_token_len

        # If no sentences passed threshold, include first few full chunks
        if not selected_sentences:
            fallback_text = "\n\n".join(c.get("text", "") for c in chunks[:3])
            return fallback_text[: self.max_tokens * 4]

        # Assemble compressed context
        compressed_text = " ".join(s[0] for s in selected_sentences)
        return compressed_text
