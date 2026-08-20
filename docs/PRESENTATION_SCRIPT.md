# 5-Minute Video Presentation Script & Demo Outline

**Title:** Senior AI Engineer Demo: Production Azure AI RAG Architecture & Evaluation  
**Target Time:** 5 Minutes (300 Seconds)

---

## 🎬 Video Agenda & Timestamps

```
[00:00 - 00:45] 1. Architecture & Azure Stack Overview
[00:45 - 01:45] 2. Live Chatbot & Baseline vs. Improved RAG Demo
[01:45 - 03:15] 3. RAG Failure Scenarios & Diagnostic Walkthrough
[03:15 - 04:15] 4. Quantitative Evaluation Results (Before vs. After)
[04:15 - 05:00] 5. Production Readiness & Next Steps
```

---

## 📜 Full Script & Presentation Plan

### 1. Architecture & Azure Stack Overview (00:00 - 00:45)
- **Speaker Script:**  
  *"Hello! I'm presenting our Enterprise RAG Knowledge Assistant built on the Microsoft Azure AI stack. Our production architecture ingests enterprise documents into Azure Blob Storage, triggers automated chunking via Azure Functions, generates vectors with Azure OpenAI's `text-embedding-3-large`, and indexes them into Azure AI Search using Hybrid Vector + BM25 search with Semantic Reranking. Security is enforced via Entra ID RBAC claims passed directly to OData retrieval filters."*

### 2. Live Chatbot & Baseline vs. Improved RAG Demo (00:45 - 01:45)
- **Demo Action:** Screen share Streamlit `app.py`. Show interactive chat.
- **Speaker Script:**  
  *"Let's look at the live application. We have built an interactive baseline comparison engine. When a user asks a question, the assistant retrieves context, provides grounded answers, and highlights exact citations. Below each response, our Deep Diagnostic Trace reveals latency, groundedness score, query rewrites, and rerank scores."*

### 3. RAG Failure Scenarios & Diagnostic Walkthrough (01:45 - 03:15)
- **Demo Action:** Open "Failure Scenario Playground" tab.
- **Speaker Script:**  
  *"Let's test two key failure scenarios:*  
  *First, **Scenario 3 (Version Conflicts)**: We have `Leave_Policy_2024.pdf` offering 20 days paid leave and `Leave_Policy_2026.pdf` offering 25 days. Baseline RAG returns the outdated 2024 chunk. Our Improved Engine extracts document version metadata and temporal dates to prioritize active 2026 policies.*  
  *Second, **Scenario 4 (Anti-Hallucination)**: When asked about pet insurance—a policy not in our documents—Baseline invents an answer. Our Improved Engine calculates a confidence score and triggers an Anti-Hallucination Guardrail, declining to answer rather than hallucinating."*

### 4. Quantitative Evaluation Results (03:15 - 04:15)
- **Demo Action:** Open "Evaluation Benchmark" tab.
- **Speaker Script:**  
  *"We built an automated evaluation benchmark comparing Baseline vs Improved RAG across 8 comprehensive enterprise test cases:*  
  *- **Retrieval Hit Rate** increased from **50.0% to 100.0%** (+50%).*  
  *- **Groundedness Score** improved from **44.0% to 92.0%** (+48%).*  
  *- **Citation Accuracy** improved from **37.5% to 100.0%**.*  
  *- **Hallucination Rate** dropped from **37.5% to 0.0%**.*"

### 5. Production Readiness & Conclusion (04:15 - 05:00)
- **Speaker Script:**  
  *"Before deploying to full production at 10 million documents scale, we would introduce Azure Cache for Redis Enterprise for semantic caching, deploy APIM rate limiting, and enable Azure AI Foundry continuous monitoring. Thank you!"*
