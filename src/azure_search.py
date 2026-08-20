import numpy as np
from typing import List, Dict, Any, Optional
from src.config import config
from src.chunking import Chunk

class HybridSearchEngine:
    """
    Hybrid Search Engine featuring Azure AI Search integration with an embedded 
    local Hybrid Search (Vector + BM25 + RRF + Metadata Filtering + Semantic Reranker) fallback.
    """
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: List[np.ndarray] = []
        self.bm25 = None
        self.use_azure = bool(config.AZURE_SEARCH_KEY and "your-azure" not in config.AZURE_SEARCH_ENDPOINT)

        if self.use_azure:
            try:
                from azure.core.credentials import AzureKeyCredential
                from azure.search.documents import SearchClient
                self.azure_client = SearchClient(
                    endpoint=config.AZURE_SEARCH_ENDPOINT,
                    index_name=config.AZURE_SEARCH_INDEX,
                    credential=AzureKeyCredential(config.AZURE_SEARCH_KEY)
                )
            except Exception:
                self.use_azure = False

    def index_chunks(self, chunks: List[Chunk], vectors: List[List[float]]):
        self.chunks = chunks
        self.vectors = [np.array(v, dtype=np.float32) for v in vectors]
        
        # Build local BM25 index
        try:
            from rank_bm25 import BM25Okapi
            corpus = [c.content.lower().split() for c in chunks]
            self.bm25 = BM25Okapi(corpus)
        except Exception:
            self.bm25 = None

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        use_parent_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (Vector + BM25), metadata filtering, and semantic reranking.
        """
        if not self.chunks:
            return []

        filtered_indices = []
        for i, c in enumerate(self.chunks):
            match = True
            if filters:
                if "department" in filters and filters["department"] and filters["department"] != "All":
                    # Department access control check
                    user_dept = filters["department"]
                    c_dept = c.department
                    if c_dept != "All" and user_dept.lower() not in c_dept.lower() and c_dept.lower() not in user_dept.lower():
                        match = False
                if "version" in filters and filters["version"]:
                    if c.version != filters["version"]:
                        match = False
                if "tier" in filters and filters["tier"]:
                    if c.tier != "All" and c.tier.lower() != filters["tier"].lower():
                        match = False
                if "min_date" in filters and filters["min_date"]:
                    if c.effective_date < filters["min_date"]:
                        match = False
            if match:
                filtered_indices.append(i)

        if not filtered_indices:
            return []

        # Vector scores computation
        q_vec = np.array(query_vector, dtype=np.float32)
        vec_norm = np.linalg.norm(q_vec) + 1e-9
        
        vector_scores = []
        for i in filtered_indices:
            v = self.vectors[i]
            c_norm = np.linalg.norm(v) + 1e-9
            score = float(np.dot(q_vec, v) / (vec_norm * c_norm))
            vector_scores.append((i, score))

        # Sort vector rank
        vector_scores.sort(key=lambda x: x[1], reverse=True)
        vector_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(vector_scores)}

        # BM25 scores computation
        bm25_rank = {}
        if use_hybrid and self.bm25:
            q_tokens = query.lower().split()
            bm25_scores_raw = self.bm25.get_scores(q_tokens)
            bm25_filtered = [(i, float(bm25_scores_raw[i])) for i in filtered_indices]
            bm25_filtered.sort(key=lambda x: x[1], reverse=True)
            bm25_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_filtered)}

        # Reciprocal Rank Fusion (RRF) calculation
        rrf_scores = []
        k = 60.0
        for i in filtered_indices:
            r_vec = vector_rank.get(i, 999)
            r_bm25 = bm25_rank.get(i, 999) if use_hybrid else 999
            
            if use_hybrid:
                rrf_score = (1.0 / (k + r_vec)) + (1.0 / (k + r_bm25))
            else:
                rrf_score = 1.0 / (k + r_vec)

            # Apply temporal decay / recency weighting for versioned docs
            chunk = self.chunks[i]
            if chunk.version == "2026":
                rrf_score *= 1.25  # Boost latest policy
            elif chunk.version == "2024":
                rrf_score *= 0.85  # Demote outdated policy unless specifically requested

            rrf_scores.append((i, rrf_score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = rrf_scores[:top_k * 2]

        # Semantic Reranker simulation / Cross-Encoder Scoring
        results = []
        for idx, score in top_candidates:
            chunk = self.chunks[idx]
            final_content = chunk.parent_content if use_parent_content and chunk.parent_content else chunk.content
            
            # Simple keyword overlap reranking boost
            query_words = set(query.lower().split())
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words.intersection(chunk_words)) / (len(query_words) + 1e-5)
            
            rerank_score = score * (1.0 + 0.5 * overlap) if use_reranker else score

            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "header": chunk.header,
                "content": final_content,
                "raw_content": chunk.content,
                "effective_date": chunk.effective_date,
                "version": chunk.version,
                "department": chunk.department,
                "tier": chunk.tier,
                "score": round(float(rerank_score), 4),
                "chunk_index": chunk.chunk_index
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
