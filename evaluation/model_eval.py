"""
Model-Level Evaluation:
Compares model backends (Gemini, vLLM, SGLang), measures consistency across runs,
and analyzes relative latency and throughput trade-offs.
"""

from typing import Dict, Any, List
from llm.engine_factory import get_llm_engine
from rag.scoring import compute_cosine_similarity
from rag.embeddings import get_embedding_provider


class ModelEvaluator:
    def __init__(self):
        self.embed_provider = get_embedding_provider()

    def compare_models(
        self,
        prompt: str,
        system_prompt: str,
        models: List[str] = ["gemini", "vllm", "sglang"]
    ) -> Dict[str, Any]:
        """
        Executes query on multiple model engines and compares latency, TPS, and output variance.
        """
        results = {}
        for m in models:
            engine = get_llm_engine(backend=m)
            text, telemetry = engine.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=512,
                temperature=0.2
            )
            results[m] = {
                "response_sample": text[:150] + "..." if len(text) > 150 else text,
                "latency_ms": round(telemetry.get("total_latency_ms", 0.0), 2),
                "ttft_ms": round(telemetry.get("ttft_ms", 0.0), 2),
                "tokens_generated": telemetry.get("tokens_generated", 0),
                "tokens_per_second": telemetry.get("tokens_per_second", 0.0)
            }

        return {"model_comparisons": results}

    def measure_consistency(
        self,
        prompt: str,
        system_prompt: str,
        backend: str = "gemini",
        runs: int = 3
    ) -> Dict[str, Any]:
        """
        Runs multiple generations with temperature > 0 to evaluate output stability/drift.
        """
        engine = get_llm_engine(backend=backend)
        outputs = []
        embeddings = []

        for _ in range(runs):
            text, _ = engine.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.3)
            outputs.append(text)
            embeddings.append(self.embed_provider.embed_text(text))

        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = compute_cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        avg_consistency = sum(similarities) / max(len(similarities), 1) if similarities else 1.0

        return {
            "runs_evaluated": runs,
            "average_consistency_score": round(avg_consistency, 3),
            "stability_rating": "High" if avg_consistency > 0.85 else "Moderate" if avg_consistency > 0.70 else "Low"
        }
