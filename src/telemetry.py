import time
import json
from typing import List, Dict, Any, Optional

class TelemetryLogger:
    """
    Production Telemetry Logger simulating Azure Application Insights distributed tracing.
    Records request spans, retrieval scores (@search.score, @search.reranker_score),
    stage latency, token usage, cost estimation, and groundedness guardrails.
    """
    def __init__(self):
        self.traces: List[Dict[str, Any]] = []

    def log_request(
        self,
        request_id: str,
        user_query: str,
        rewritten_query: str,
        user_department: str,
        engine: str,
        search_scores: List[Dict[str, float]],
        retrieval_latency_ms: float,
        llm_latency_ms: float,
        total_latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
        groundedness_score: float,
        confidence_score: float,
        is_ambiguous: bool = False,
        insufficient_evidence: bool = False,
        error: Optional[str] = None
    ):
        trace = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request_id": request_id,
            "engine": engine,
            "user_query": user_query,
            "rewritten_query": rewritten_query,
            "user_department": user_department,
            "retrieval": {
                "latency_ms": round(retrieval_latency_ms, 2),
                "chunks_retrieved": len(search_scores),
                "scores": search_scores
            },
            "generation": {
                "latency_ms": round(llm_latency_ms, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": round(estimated_cost_usd, 6)
            },
            "metrics": {
                "total_latency_ms": round(total_latency_ms, 2),
                "groundedness_score": round(groundedness_score, 2),
                "confidence_score": round(confidence_score, 2),
                "is_ambiguous": is_ambiguous,
                "insufficient_evidence": insufficient_evidence
            },
            "error": error
        }
        self.traces.append(trace)
        return trace

    def get_latest_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.traces[-limit:]

telemetry = TelemetryLogger()
