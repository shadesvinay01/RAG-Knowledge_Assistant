import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.config import config
from src.chunking import HierarchicalChunker
from src.embeddings import EmbeddingEngine
from src.azure_search import HybridSearchEngine
from src.rag_engine import BaselineRAGEngine, ImprovedEnterpriseRAGEngine
from eval.evaluate import load_knowledge_base, run_evaluation

# Page Configuration
st.set_page_config(
    page_title="Enterprise Azure RAG Knowledge Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0078D4; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #505050; margin-bottom: 20px; }
    .stAlert { border-radius: 8px; }
    .chunk-card { background-color: #F0F4F8; padding: 12px; border-radius: 8px; border-left: 4px solid #0078D4; margin-bottom: 10px; }
    .metric-card { background: linear-gradient(135deg, #0078D4 0%, #002050 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State & Indexes
@st.cache_resource
def init_rag_system():
    data_dir = "data/knowledge_base"
    chunks = load_knowledge_base(data_dir)
    embedding_engine = EmbeddingEngine()
    chunk_texts = [c.content for c in chunks]
    vectors = embedding_engine.embed_texts(chunk_texts)

    search_engine = HybridSearchEngine()
    search_engine.index_chunks(chunks, vectors)

    baseline_engine = BaselineRAGEngine(search_engine, embedding_engine)
    improved_engine = ImprovedEnterpriseRAGEngine(search_engine, embedding_engine)

    return chunks, search_engine, baseline_engine, improved_engine

chunks, search_engine, baseline_engine, improved_engine = init_rag_system()

# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Label.svg", width=180)
    st.title("⚙️ RAG Settings")

    engine_choice = st.radio(
        "Select RAG Pipeline:",
        ["Improved Enterprise RAG (Production)", "Baseline RAG (Naive)"],
        index=0
    )

    st.markdown("---")
    st.subheader("👤 User Security Scoping (RBAC)")
    user_dept = st.selectbox("Department Access Level:", ["All", "Engineering", "HR", "Finance", "Legal"], index=0)
    
    st.markdown("---")
    st.subheader("📅 Temporal Filter")
    policy_version = st.selectbox("Policy Version Override:", ["Auto / Latest (2026)", "2026", "2024"], index=0)
    version_filter = None if "Auto" in policy_version else policy_version

    st.markdown("---")
    bypass_cache = st.checkbox("Bypass Semantic Cache", value=False)
    
    st.markdown("---")
    st.info("💡 **Azure Mode Active:** Native Azure OpenAI & Azure AI Search SDK connected with local fallback.")

# App Navigation Tabs
tab_chat, tab_scenarios, tab_eval, tab_arch = st.tabs([
    "💬 Knowledge Assistant",
    "🧪 Failure Scenario Playground",
    "📊 Evaluation Benchmark",
    "🏗️ Enterprise Architecture"
])

# ==========================================
# TAB 1: INTERACTIVE CHAT & INSPECTOR
# ==========================================
with tab_chat:
    st.markdown('<div class="main-header">Enterprise Knowledge Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Azure AI RAG Stack with Grounded Citations & Failure Diagnosis</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Clear chat button
    col_c1, col_c2 = st.columns([6, 1])
    with col_c2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Display Chat Messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                st.caption("📌 **Citations:** " + ", ".join(msg["citations"]))

    # Chat Input
    if user_query := st.chat_input("Ask a question about HR policies, Enterprise vs Standard refunds, or system limits..."):
        # User message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Bot Response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving grounded context & reasoning..."):
                # Prepare past conversation history formatted for query rewriter (Scenario 6)
                formatted_history = []
                for i in range(0, len(st.session_state.chat_history) - 1, 2):
                    if i + 1 < len(st.session_state.chat_history):
                        formatted_history.append({
                            "user": st.session_state.chat_history[i]["content"],
                            "assistant": st.session_state.chat_history[i+1]["content"]
                        })

                if "Baseline" in engine_choice:
                    res = baseline_engine.query(user_query)
                else:
                    res = improved_engine.query(
                        user_query=user_query,
                        chat_history=formatted_history,
                        user_department=user_dept,
                        requested_version=version_filter,
                        bypass_cache=bypass_cache
                    )

                st.markdown(res["answer"])
                if res["citations"]:
                    st.caption("📌 **Citations:** " + ", ".join(res["citations"]))

                # Diagnostic Inspector
                with st.expander("🔍 Deep Diagnostic Trace & Retrieval Inspector"):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Latency", f"{res['latency_ms']} ms")
                    col2.metric("Groundedness", f"{res['groundedness_score'] * 100:.0f}%")
                    col3.metric("Confidence Score", f"{res.get('confidence_score', 1.0) * 100:.0f}%")
                    col4.metric("Tokens Used", res["tokens_used"])

                    if res.get("rewritten_query"):
                        st.markdown(f"**Query Rewriter (Scenario 6):** `{res['rewritten_query']}`")
                    if res.get("sub_queries"):
                        st.markdown(f"**Decomposed Sub-Queries (Scenario 2):** `{res['sub_queries']}`")
                    if res.get("is_ambiguous"):
                        st.warning("⚠️ **Ambiguity Guardrail Triggered (Scenario 5):** Clarification requested from user.")
                    if res.get("insufficient_evidence"):
                        st.error("🛡️ **Anti-Hallucination Guardrail Triggered (Scenario 4):** Confidence score below threshold.")

                    st.markdown("#### Retrieved Chunks & Rerank Scores")
                    for chunk in res["retrieved_chunks"]:
                        st.markdown(f"""
                        <div class="chunk-card">
                            <strong>📄 {chunk['doc_name']}</strong> ({chunk['header']}) | 
                            <em>Version: {chunk['version']} | Score: {chunk['score']}</em><br/>
                            {chunk['content']}
                        </div>
                        """, unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "citations": res["citations"]
                })

# ==========================================
# TAB 2: FAILURE SCENARIOS PLAYGROUND
# ==========================================
with tab_scenarios:
    st.header("🧪 RAG Failure Scenarios Playground")
    st.markdown("Test the 6 specific enterprise RAG failure scenarios side-by-side to observe Baseline failure vs. Improved RAG resolution.")

    scenarios = {
        "Scenario 1: Correct Document, Wrong Chunk": {
            "query": "How many days of paid sick leave do employees get under the 2026 policy and when is a doctor certificate needed?",
            "desc": "Baseline returns an unrelated header chunk. Improved RAG uses Parent-Child Linkage & Semantic Reranking to deliver exact context."
        },
        "Scenario 2: Information Across Multiple Sections": {
            "query": "Compare the refund policy for Enterprise and Standard customers.",
            "desc": "Requires combining information from two distinct markdown documents. Improved RAG uses Multi-Query Decomposition."
        },
        "Scenario 3: Similar Documents / Conflicting Information": {
            "query": "How many days of annual paid leave are employees entitled to?",
            "desc": "Knowledge base has Leave_Policy_2024.pdf (20 days) and Leave_Policy_2026.pdf (25 days). Improved RAG applies version metadata filtering."
        },
        "Scenario 4: Hallucination / Missing Information": {
            "query": "What is the company policy on pet insurance reimbursement for remote employees?",
            "desc": "Answer does not exist in knowledge base. Baseline hallucinates; Improved RAG enforces Groundedness Guardrail threshold."
        },
        "Scenario 5: Ambiguous Query": {
            "query": "What is the limit?",
            "desc": "Query is underspecified. Baseline retrieves random limit; Improved RAG detects ambiguity and prompts for clarification."
        },
        "Scenario 6: Conversational Context": {
            "query": "What about Standard?",
            "history": [{"user": "What is the Enterprise plan cancellation policy?", "assistant": "Enterprise plans require 30-day notice."}],
            "desc": "Follow-up question without explicit context. Improved RAG rewrites query to avoid retrieval pollution."
        }
    }

    selected_scen = st.selectbox("Select RAG Failure Scenario to Test:", list(scenarios.keys()))
    scen_data = scenarios[selected_scen]
    st.info(f"**Test Description:** {scen_data['desc']}")
    st.markdown(f"**Test Query:** `{scen_data['query']}`")

    if st.button("▶️ Run Side-by-Side Comparison"):
        col_b, col_i = st.columns(2)

        with col_b:
            st.subheader("🔴 Baseline RAG Result")
            with st.spinner("Running Baseline..."):
                b_res = baseline_engine.query(scen_data["query"])
                st.write(b_res["answer"])
                st.caption("Citations: " + ", ".join(b_res["citations"]))
                st.caption(f"Latency: {b_res['latency_ms']} ms | Groundedness: {b_res['groundedness_score']*100:.0f}%")

        with col_i:
            st.subheader("🟢 Improved Enterprise RAG Result")
            with st.spinner("Running Improved..."):
                i_res = improved_engine.query(
                    user_query=scen_data["query"],
                    chat_history=scen_data.get("history"),
                    bypass_cache=True
                )
                st.write(i_res["answer"])
                st.caption("Citations: " + ", ".join(i_res["citations"]))
                st.caption(f"Latency: {i_res['latency_ms']} ms | Groundedness: {i_res['groundedness_score']*100:.0f}%")

# ==========================================
# TAB 3: BENCHMARK EVALUATION DASHBOARD
# ==========================================
with tab_eval:
    st.header("📊 Automated Evaluation Benchmark Dashboard")
    st.markdown("Quantitative benchmark metrics comparing **Baseline RAG** vs **Improved Enterprise RAG**.")

    if st.button("🚀 Run Full Automated Benchmark"):
        with st.spinner("Evaluating test dataset across retrieval & generation metrics..."):
            report = run_evaluation()

    if os.path.exists("eval/eval_results.json"):
        with open("eval/eval_results.json", "r") as f:
            eval_data = json.load(f)
        
        s = eval_data["summary"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieval Hit Rate @ K", f"{s['improved_hit_rate']*100:.1f}%", f"+{(s['improved_hit_rate']-s['baseline_hit_rate'])*100:.1f}%")
        m2.metric("Groundedness Score", f"{s['improved_groundedness']*100:.1f}%", f"+{(s['improved_groundedness']-s['baseline_groundedness'])*100:.1f}%")
        m3.metric("Citation Accuracy", f"{s['improved_citation_acc']*100:.1f}%", f"+{(s['improved_citation_acc']-s['baseline_citation_acc'])*100:.1f}%")
        m4.metric("Hallucination Rate", f"{s['improved_hallucination_rate']*100:.1f}%", f"-{(s['baseline_hallucination_rate']-s['improved_hallucination_rate'])*100:.1f}%", delta_color="inverse")

        st.markdown("---")
        df_chart = pd.DataFrame({
            "Metric": ["Retrieval Hit Rate @ K", "Groundedness Score", "Citation Accuracy", "Hallucination Rate"],
            "Baseline RAG": [s["baseline_hit_rate"]*100, s["baseline_groundedness"]*100, s["baseline_citation_acc"]*100, s["baseline_hallucination_rate"]*100],
            "Improved Enterprise RAG": [s["improved_hit_rate"]*100, s["improved_groundedness"]*100, s["improved_citation_acc"]*100, s["improved_hallucination_rate"]*100]
        })


        fig = px.bar(df_chart, x="Metric", y=["Baseline RAG", "Improved Enterprise RAG"], barmode="group",
                     title="Baseline vs. Improved RAG Accuracy Metrics (%)",
                     color_discrete_sequence=["#E74C3C", "#2ECC71"])
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# TAB 4: ENTERPRISE ARCHITECTURE
# ==========================================
with tab_arch:
    st.header("🏗️ Enterprise Azure AI Production Architecture")
    st.markdown("""
    ### End-to-End Enterprise RAG Architecture
    ```mermaid
    graph TD
        A[Documents: PDFs, MDs] --> B[Azure Blob Storage]
        B --> C[Azure Functions Ingestion]
        C --> D[Hierarchical Chunker]
        D --> E[Azure OpenAI Embeddings text-embedding-3-large]
        E --> F[Azure AI Search: Hybrid Vector + BM25 + Semantic Ranker]
        
        G[User Query / App] --> H[Azure API Management / App Service]
        H --> I[Entra ID Security RBAC Claims]
        I --> J[Query Rewriter & Ambiguity Detector]
        J --> F
        F --> K[Context Aggregator & Parent Chunks]
        K --> L[Azure OpenAI gpt-4o Grounded Generation]
        L --> M[Anti-Hallucination Guardrail]
        M --> N[Grounded Answer + Citations]
    ```
    """)
    st.markdown("""
    #### Architectural Pillars & Rationale:
    1. **Azure AI Search (Hybrid + Semantic Ranker):** Provides combined vector search and BM25 text search with reciprocal rank fusion (RRF) and deep transformer reranking.
    2. **Security Isolation (Entra ID & OData Filters):** RBAC claims are passed as OData filters during retrieval so users only retrieve documents authorized for their department.
    3. **Azure Key Vault & Secrets Management:** API keys, storage connections, and database credentials stored securely in Key Vault.
    4. **Application Insights Observability:** Telemetry logging for end-to-end tracing, latency breakdown, and token usage cost tracking.
    """)
