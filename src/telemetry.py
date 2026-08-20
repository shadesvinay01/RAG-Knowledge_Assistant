import os
import time
import json
from typing import List, Dict, Any, Optional
from src.config import config

class TelemetryLogger:
    """
    Production Telemetry Logger featuring Azure Application Insights OpenTelemetry SDK integration.
    Exports request traces, search scores (@search.score, @search.reranker_score),
    stage latency, token usage, cost estimation, and groundedness guardrails.
    """
    def __init__(self):
        self.traces: List[Dict[str, Any]] = []
        self.use_app_insights = bool(config.APPLICATIONINSIGHTS_CONNECTION_STRING)
        self.tracer = None

        if self.use_app_insights:
            try:
                from azure.monitor.opentelemetry import configure_azure_monitor
                from opentelemetry import trace
                configure_azure_monitor(connection_string=config.APPLICATIONINSIGHTS_CONNECTION_STRING)
                self.tracer = trace.get_tracer("enterprise-rag-tracer")
                print("✅ Azure Application Insights OpenTelemetry Exporter initialized.")
            except Exception as e:
                print(f"[Warning] Application Insights SDK initialization skipped ({e}). Storing local trace list.")
                self.use_app_insights = False

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

        # Export trace to Azure Application Insights if active
        if self.use_app_insights and self.tracer:
            try:
                with self.tracer.start_as_current_span("rag_request_span") as span:
                    span.set_attribute("request_id", request_id)
                    span.set_attribute("user_query", user_query)
                    span.set_attribute("engine", engine)
                    span.set_attribute("total_latency_ms", total_latency_ms)
                    span.set_attribute("estimated_cost_usd", estimated_cost_usd)
                    span.set_attribute("groundedness_score", groundedness_score)
            except Exception as e:
                print(f"[Warning] Failed to export span to App Insights: {e}")

        self.traces.append(trace)
        return trace

    def get_latest_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.traces[-limit:]

telemetry = TelemetryLogger()
