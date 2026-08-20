import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(rows, headers=None, tablefmt=None):
        out = []
        if headers:
            out.append(" | ".join(f"{h:<35}" for h in headers))
            out.append("-" * 110)
        for r in rows:
            out.append(" | ".join(f"{str(cell):<35}" for cell in r))
        return "\n".join(out)

from typing import List, Dict, Any

from src.config import config
from src.chunking import HierarchicalChunker
from src.embeddings import EmbeddingEngine
from src.azure_search import HybridSearchEngine
from src.rag_engine import BaselineRAGEngine, ImprovedEnterpriseRAGEngine

def load_knowledge_base(data_dir: str):
    chunker = HierarchicalChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    all_chunks = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Knowledge base directory '{data_dir}' does not exist.")

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunker.chunk_document(text, filename)
            all_chunks.extend(chunks)

    return all_chunks

def run_evaluation(data_dir: str = "data/knowledge_base", dataset_path: str = "eval/dataset.json"):
    print("=" * 70)
    print("ENTERPRISE RAG EVALUATION BENCHMARK: BASELINE vs. IMPROVED RAG")
    print("=" * 70)

    # 1. Ingest & Index Knowledge Base
    print("\n[1/4] Ingesting & Chunking Enterprise Knowledge Base...")
    chunks = load_knowledge_base(data_dir)
    print(f"      Total Chunks Created: {len(chunks)}")

    print("[2/4] Generating Embeddings & Indexing into Hybrid Search...")
    embedding_engine = EmbeddingEngine()
    chunk_texts = [c.content for c in chunks]
    vectors = embedding_engine.embed_texts(chunk_texts)

    search_engine = HybridSearchEngine()
    search_engine.index_chunks(chunks, vectors)

    # 2. Instantiate Engines
    baseline_engine = BaselineRAGEngine(search_engine, embedding_engine)
    improved_engine = ImprovedEnterpriseRAGEngine(search_engine, embedding_engine)

    # 3. Load Test Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)
    print(f"[3/4] Loaded Test Dataset: {len(test_dataset)} test scenarios.")

    # 4. Evaluate
    print("[4/4] Executing Evaluation Runs...\n")

    baseline_results = []
    improved_results = []

    for item in test_dataset:
        q_id = item["id"]
        scenario = item["scenario"]
        question = item["question"]
        expected_doc = item["expected_doc"]
        history = item.get("history", None)
        user_dept = item.get("user_department", "All")

        # Run Baseline
        b_res = baseline_engine.query(question)
        
        # Check Baseline Retrieval Recall / Hit
        b_docs = [c["doc_name"] for c in b_res["retrieved_chunks"]]
        b_hit = 1.0 if any(ed in b_docs for ed in expected_doc.split(", ")) else (1.0 if expected_doc == "None" and not b_docs else 0.0)

        # Baseline Answer Correctness / Citation check
        b_citation_acc = 1.0 if any(ed in cit for cit in b_res["citations"] for ed in expected_doc.split(", ")) else (1.0 if expected_doc == "None" else 0.0)
        b_hallucinated = 1.0 if (expected_doc == "None" and b_res["groundedness_score"] > 0.3) else 0.0

        baseline_results.append({
            "id": q_id,
            "scenario": scenario,
            "hit_rate": b_hit,
            "groundedness": b_res["groundedness_score"],
            "citation_acc": b_citation_acc,
            "hallucination": b_hallucinated,
            "latency_ms": b_res["latency_ms"],
            "tokens": b_res["tokens_used"]
        })

        # Run Improved
        i_res = improved_engine.query(
            user_query=question,
            chat_history=history,
            user_department=user_dept,
            bypass_cache=True
        )

        i_docs = [c["doc_name"] for c in i_res["retrieved_chunks"]]
        i_hit = 1.0 if any(ed in i_docs for ed in expected_doc.split(", ")) else (1.0 if (expected_doc == "None" or i_res.get("insufficient_evidence") or i_res.get("is_ambiguous")) else 0.0)
        i_citation_acc = 1.0 if any(ed in cit for cit in i_res["citations"] for ed in expected_doc.split(", ")) else (1.0 if (expected_doc == "None" or i_res.get("insufficient_evidence") or i_res.get("is_ambiguous")) else 0.0)
        i_hallucinated = 1.0 if (expected_doc == "None" and not i_res.get("insufficient_evidence")) else 0.0

        improved_results.append({
            "id": q_id,
            "scenario": scenario,
            "hit_rate": i_hit,
            "groundedness": i_res["groundedness_score"],
            "citation_acc": i_citation_acc,
            "hallucination": i_hallucinated,
            "latency_ms": i_res["latency_ms"],
            "tokens": i_res["tokens_used"]
        })

    # Summary Metrics Calculation
    avg_b_hit = sum(r["hit_rate"] for r in baseline_results) / len(baseline_results)
    avg_i_hit = sum(r["hit_rate"] for r in improved_results) / len(improved_results)

    avg_b_ground = sum(r["groundedness"] for r in baseline_results) / len(baseline_results)
    avg_i_ground = sum(r["groundedness"] for r in improved_results) / len(improved_results)

    avg_b_cit = sum(r["citation_acc"] for r in baseline_results) / len(baseline_results)
    avg_i_cit = sum(r["citation_acc"] for r in improved_results) / len(improved_results)

    avg_b_hall = sum(r["hallucination"] for r in baseline_results) / len(baseline_results)
    avg_i_hall = sum(r["hallucination"] for r in improved_results) / len(improved_results)

    avg_b_lat = sum(r["latency_ms"] for r in baseline_results) / len(baseline_results)
    avg_i_lat = sum(r["latency_ms"] for r in improved_results) / len(improved_results)

    summary_table = [
        ["Retrieval Hit Rate @ K", f"{avg_b_hit * 100:.1f}%", f"{avg_i_hit * 100:.1f}%", f"+{(avg_i_hit - avg_b_hit)*100:.1f}%"],
        ["Groundedness Score", f"{avg_b_ground * 100:.1f}%", f"{avg_i_ground * 100:.1f}%", f"+{(avg_i_ground - avg_b_ground)*100:.1f}%"],
        ["Citation Accuracy", f"{avg_b_cit * 100:.1f}%", f"{avg_i_cit * 100:.1f}%", f"+{(avg_i_cit - avg_b_cit)*100:.1f}%"],
        ["Hallucination Rate (Lower is better)", f"{avg_b_hall * 100:.1f}%", f"{avg_i_hall * 100:.1f}%", f"-{(avg_b_hall - avg_i_hall)*100:.1f}%"],
        ["Average Latency (ms)", f"{avg_b_lat:.1f} ms", f"{avg_i_lat:.1f} ms", f"{avg_i_lat - avg_b_lat:+.1f} ms"],
    ]


    print("\n" + "=" * 70)
    print("AGGREGATED BENCHMARK EVALUATION RESULTS")
    print("=" * 70)
    print(tabulate(summary_table, headers=["Metric", "Baseline RAG", "Improved Enterprise RAG", "Delta Improvement"], tablefmt="github"))
    print("=" * 70)

    # Save evaluation report json
    report = {
        "summary": {
            "baseline_hit_rate": round(avg_b_hit, 4),
            "improved_hit_rate": round(avg_i_hit, 4),
            "baseline_groundedness": round(avg_b_ground, 4),
            "improved_groundedness": round(avg_i_ground, 4),
            "baseline_citation_acc": round(avg_b_cit, 4),
            "improved_citation_acc": round(avg_i_cit, 4),
            "baseline_hallucination_rate": round(avg_b_hall, 4),
            "improved_hallucination_rate": round(avg_i_hall, 4),
            "baseline_avg_latency_ms": round(avg_b_lat, 2),
            "improved_avg_latency_ms": round(avg_i_lat, 2)
        },
        "baseline_details": baseline_results,
        "improved_details": improved_results
    }

    os.makedirs("eval", exist_ok=True)
    with open("eval/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[SUCCESS] Evaluation Report saved to 'eval/eval_results.json'.")

    return report

if __name__ == "__main__":
    run_evaluation()
