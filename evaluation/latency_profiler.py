"""
Latency and Throughput Profiler:
Analyzes average latency speed, TTFT (Time to First Token), tokens/sec throughput,
and p50/p95 latency percentiles across Simple, Hybrid, and Graph RAG pipelines.
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from database import SessionLocal, QueryLogModel


class LatencyProfiler:
    @staticmethod
    def get_summary_statistics() -> Dict[str, Any]:
        """
        Queries historical database query logs to calculate:
        - Average total latency (ms)
        - Average retrieval latency (ms)
        - Average generation latency (ms)
        - P50, P95, P99 latency percentiles
        - Average Tokens Per Second (TPS)
        - Cache hit ratio
        """
        try:
            with SessionLocal() as db:
                logs = db.query(QueryLogModel).all()
                if not logs:
                    return {
                        "total_queries_logged": 0,
                        "avg_total_latency_ms": 0.0,
                        "p95_latency_ms": 0.0,
                        "avg_retrieval_latency_ms": 0.0,
                        "avg_generation_latency_ms": 0.0,
                        "avg_tokens_per_second": 0.0,
                        "cache_hit_rate": 0.0
                    }

                total_latencies = [l.total_latency_ms for l in logs if l.total_latency_ms > 0]
                retrieval_latencies = [l.retrieval_latency_ms for l in logs]
                gen_latencies = [l.generation_latency_ms for l in logs]
                tps_values = [l.tokens_per_second for l in logs if l.tokens_per_second > 0]
                cache_hits = sum(1 for l in logs if l.cache_hit)

                p50 = float(np.percentile(total_latencies, 50)) if total_latencies else 0.0
                p95 = float(np.percentile(total_latencies, 95)) if total_latencies else 0.0
                p99 = float(np.percentile(total_latencies, 99)) if total_latencies else 0.0

                return {
                    "total_queries_logged": len(logs),
                    "avg_total_latency_ms": round(float(np.mean(total_latencies)), 2) if total_latencies else 0.0,
                    "p50_latency_ms": round(p50, 2),
                    "p95_latency_ms": round(p95, 2),
                    "p99_latency_ms": round(p99, 2),
                    "avg_retrieval_latency_ms": round(float(np.mean(retrieval_latencies)), 2) if retrieval_latencies else 0.0,
                    "avg_generation_latency_ms": round(float(np.mean(gen_latencies)), 2) if gen_latencies else 0.0,
                    "avg_tokens_per_second": round(float(np.mean(tps_values)), 2) if tps_values else 0.0,
                    "cache_hit_rate": round(cache_hits / len(logs), 3) if logs else 0.0
                }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def benchmark_pipeline(pipeline_fn, question: str, iterations: int = 3) -> Dict[str, Any]:
        """Runs iterative benchmark on a specific pipeline function."""
        durations = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            res = pipeline_fn(question)
            t1 = time.perf_counter()
            durations.append((t1 - t0) * 1000)

        return {
            "iterations": iterations,
            "min_latency_ms": round(min(durations), 2),
            "max_latency_ms": round(max(durations), 2),
            "avg_latency_ms": round(float(np.mean(durations)), 2),
            "std_dev_ms": round(float(np.std(durations)), 2)
        }
