"""
Unified Evaluation Orchestrator:
Coordinates Prompt-Level, Response-Level, Model-Level, Corruption Analysis,
and Latency Profiling, storing results to the database.
"""

from typing import Dict, Any, List, Optional
from database import SessionLocal, EvaluationLogModel
from .prompt_eval import PromptEvaluator
from .response_eval import ResponseEvaluator
from .model_eval import ModelEvaluator
from .corruption import CorruptionAnalyzer
from .latency_profiler import LatencyProfiler


class RAGEvaluator:
    def __init__(self):
        self.prompt_evaluator = PromptEvaluator()
        self.response_evaluator = ResponseEvaluator()
        self.model_evaluator = ModelEvaluator()
        self.corruption_analyzer = CorruptionAnalyzer()
        self.latency_profiler = LatencyProfiler()

    def evaluate_turn(
        self,
        query: str,
        response: str,
        context_passages: List[str],
        rag_type: str = "hybrid",
        language: str = "en",
        mode: str = "general",
        query_id: Optional[int] = None,
        include_corruption_test: bool = False
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation across Prompt, Response, and Corruption dimensions.
        """
        # 1. Prompt-level evaluation
        p_metrics = self.prompt_evaluator.evaluate(prompt=query)

        # 2. Response-level evaluation
        r_metrics = self.response_evaluator.evaluate(
            query=query,
            response=response,
            context_passages=context_passages,
            expected_mode=mode,
            expected_language=language
        )

        # 3. Optional Corruption analysis
        corruption_metrics = None
        if include_corruption_test and context_passages:
            combined_ctx = " ".join(context_passages)
            corruption_metrics = self.corruption_analyzer.run_stress_test(
                query=query,
                clean_context=combined_ctx
            )

        # 4. Save to Database
        try:
            with SessionLocal() as db:
                eval_log = EvaluationLogModel(
                    query_id=query_id,
                    query_text=query,
                    rag_type=rag_type,
                    prompt_clarity_score=p_metrics["clarity_score"],
                    prompt_token_efficiency=p_metrics["token_efficiency"],
                    prompt_injection_risk=p_metrics["injection_risk_score"],
                    faithfulness_score=r_metrics["faithfulness_score"],
                    answer_relevancy_score=r_metrics["answer_relevancy_score"],
                    hallucination_rate=r_metrics["hallucination_rate"],
                    format_adherence_score=r_metrics["format_adherence_score"],
                    corruption_robustness_score=corruption_metrics["robustness_score"] if corruption_metrics else 1.0,
                    feedback_notes="Automated evaluation completed."
                )
                db.add(eval_log)
                db.commit()
                db.refresh(eval_log)
                log_id = eval_log.id
        except Exception as e:
            print(f"[RAGEvaluator] Notice saving evaluation log: {e}")
            log_id = None

        return {
            "evaluation_id": log_id,
            "prompt_level": p_metrics,
            "response_level": r_metrics,
            "corruption_analysis": corruption_metrics,
            "overall_health_score": round(
                (p_metrics["clarity_score"] * 0.2) +
                (r_metrics["faithfulness_score"] * 0.4) +
                (r_metrics["answer_relevancy_score"] * 0.4),
                2
            )
        }
