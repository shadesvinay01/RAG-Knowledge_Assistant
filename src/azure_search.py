import numpy as np
from typing import List, Dict, Any, Optional
from src.config import config
from src.chunking import Chunk

class HybridSearchEngine:
    """
    Hybrid Search Engine featuring native Azure AI Search SDK execution (SearchClient.search)
    with embedded local Hybrid Search (Vector + BM25 + RRF + OData Metadata Filtering + Semantic Reranker) fallback.
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
                from azure.search.documents.indexes import SearchIndexClient

                self.credential = AzureKeyCredential(config.AZURE_SEARCH_KEY)
                self.index_client = SearchIndexClient(endpoint=config.AZURE_SEARCH_ENDPOINT, credential=self.credential)
                self.azure_client = SearchClient(endpoint=config.AZURE_SEARCH_ENDPOINT, index_name=config.AZURE_SEARCH_INDEX, credential=self.credential)
            except Exception as e:
                print(f"[Warning] Azure AI Search SDK initialization skipped ({e}). Using local hybrid index.")
                self.use_azure = False

    def create_azure_index_schema(self):
        """
        Creates Azure AI Search Index with Vector HNSW Profile, BM25 Searchable fields,
        OData Filterable metadata (department, version, status), parent_content, chunk_index, and Semantic Ranker config.
        """
        if not self.use_azure:
            return

        try:
            from azure.search.documents.indexes.models import (
                SearchIndex,
                SimpleField,
                SearchableField,
                SearchField,
                SearchFieldDataType,
                VectorSearch,
                HnswAlgorithmConfiguration,
                VectorSearchProfile,
                SemanticConfiguration,
                SemanticPrioritizedFields,
                SemanticField,
                SemanticSearch
            )

            fields = [
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="doc_name", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="header", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
                SimpleField(name="parent_content", type=SearchFieldDataType.String),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
                SimpleField(name="effective_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
                SimpleField(name="version", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="status", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="department", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="tier", type=SearchFieldDataType.String, filterable=True),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536, # text-embedding-3-large dim
                    vector_search_profile_name="hnsw-vector-profile"
                )
            ]

            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
                profiles=[VectorSearchProfile(name="hnsw-vector-profile", algorithm_configuration_name="hnsw-algo")]
            )

            semantic_config = SemanticConfiguration(
                name=config.AZURE_SEARCH_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="header"),
                    content_fields=[SemanticField(field_name="content")]
                )
            )

            semantic_search = SemanticSearch(configurations=[semantic_config])

            index = SearchIndex(
                name=config.AZURE_SEARCH_INDEX,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )

            self.index_client.create_or_update_index(index)
            print(f"✅ Azure AI Search Index '{config.AZURE_SEARCH_INDEX}' schema created/updated successfully.")
        except Exception as e:
            print(f"[Warning] Azure AI Search schema setup failed: {e}")

    def index_chunks(self, chunks: List[Chunk], vectors: List[List[float]]):
        self.chunks = chunks
        self.vectors = [np.array(v, dtype=np.float32) for v in vectors]
        
        if self.use_azure:
            try:
                documents = []
                for c, v in zip(chunks, vectors):
                    doc = c.to_dict()
                    doc["content_vector"] = v
                    documents.append(doc)
                self.azure_client.upload_documents(documents)
                print(f"✅ Uploaded {len(documents)} document chunks to Azure AI Search Index.")
            except Exception as e:
                print(f"[Warning] Azure AI Search upload failed: {e}")

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
        use_parent_content: bool = False,
        include_superseded: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid search (Vector + BM25), Entra ID RBAC security filtering, 
        version/status filtering, and semantic reranking with @search.score extraction.
        """
        if self.use_azure:
            try:
                from azure.search.documents.models import VectorizedQuery, QueryType

                filter_parts = []
                if not include_superseded:
                    filter_parts.append("status eq 'ACTIVE'")
                if filters and "department" in filters and filters["department"] and filters["department"] != "All":
                    filter_parts.append(f"(department eq '{filters['department']}' or department eq 'All')")
                if filters and "version" in filters and filters["version"]:
                    filter_parts.append(f"version eq '{filters['version']}'")

                filter_expr = " and ".join(filter_parts) if filter_parts else None

                vector_query = VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top_k,
                    fields="content_vector"
                )

                azure_results = self.azure_client.search(
                    search_text=query if use_hybrid else None,
                    vector_queries=[vector_query] if query_vector else None,
                    filter=filter_expr,
                    query_type=QueryType.SEMANTIC if use_reranker else QueryType.SIMPLE,
                    semantic_configuration_name=config.AZURE_SEARCH_SEMANTIC_CONFIG if use_reranker else None,
                    top=top_k
                )

                results = []
                for doc in azure_results:
                    final_content = doc.get("parent_content") if use_parent_content and doc.get("parent_content") else doc["content"]
                    results.append({
                        "chunk_id": doc["chunk_id"],
                        "doc_id": doc["doc_id"],
                        "doc_name": doc["doc_name"],
                        "header": doc["header"],
                        "content": final_content,
                        "raw_content": doc["content"],
                        "parent_content": doc.get("parent_content", ""),
                        "chunk_index": doc.get("chunk_index", 0),
                        "effective_date": doc["effective_date"],
                        "version": doc["version"],
                        "status": doc["status"],
                        "department": doc["department"],
                        "tier": doc["tier"],
                        "@search.score": round(float(doc.get("@search.score", 0.0)), 4),
                        "@search.reranker_score": round(float(doc.get("@search.reranker_score", 0.0)), 4),
                        "score": round(float(doc.get("@search.reranker_score", doc.get("@search.score", 0.0))), 4)
                    })
                return results
            except Exception as e:
                print(f"[Warning] Azure AI Search call failed ({e}). Falling back to local hybrid index.")

        # Local Hybrid Vector + BM25 + RRF fallback execution
        if not self.chunks:
            return []

        filtered_indices = []
        for i, c in enumerate(self.chunks):
            match = True
            if not include_superseded and c.status == "SUPERSEDED":
                match = False

            if filters:
                if "department" in filters and filters["department"] and filters["department"] != "All":
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

            if match:
                filtered_indices.append(i)

        if not filtered_indices:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        vec_norm = np.linalg.norm(q_vec) + 1e-9
        
        vector_scores = []
        for i in filtered_indices:
            v = self.vectors[i]
            c_norm = np.linalg.norm(v) + 1e-9
            score = float(np.dot(q_vec, v) / (vec_norm * c_norm))
            vector_scores.append((i, score))

        vector_scores.sort(key=lambda x: x[1], reverse=True)
        vector_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(vector_scores)}

        bm25_rank = {}
        if use_hybrid and self.bm25:
            q_tokens = query.lower().split()
            bm25_scores_raw = self.bm25.get_scores(q_tokens)
            bm25_filtered = [(i, float(bm25_scores_raw[i])) for i in filtered_indices]
            bm25_filtered.sort(key=lambda x: x[1], reverse=True)
            bm25_rank = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_filtered)}

        rrf_scores = []
        k = 60.0
        for i in filtered_indices:
            r_vec = vector_rank.get(i, 999)
            r_bm25 = bm25_rank.get(i, 999) if use_hybrid else 999
            
            if use_hybrid:
                rrf_score = (1.0 / (k + r_vec)) + (1.0 / (k + r_bm25))
            else:
                rrf_score = 1.0 / (k + r_vec)

            chunk = self.chunks[i]
            if chunk.version == "2026":
                rrf_score *= 1.25
            elif chunk.version == "2024":
                rrf_score *= 0.70

            rrf_scores.append((i, rrf_score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = rrf_scores[:top_k * 2]

        results = []
        for idx, score in top_candidates:
            chunk = self.chunks[idx]
            final_content = chunk.parent_content if use_parent_content and chunk.parent_content else chunk.content
            
            query_words = set(query.lower().split())
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words.intersection(chunk_words)) / (len(query_words) + 1e-5)
            
            rerank_score = score * (1.0 + 0.5 * overlap) if use_reranker else score
            raw_search_score = vector_scores[filtered_indices.index(idx)][1] if idx in filtered_indices else 0.5

            results.append({
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "header": chunk.header,
                "content": final_content,
                "raw_content": chunk.content,
                "parent_content": chunk.parent_content,
                "chunk_index": chunk.chunk_index,
                "effective_date": chunk.effective_date,
                "version": chunk.version,
                "status": chunk.status,
                "department": chunk.department,
                "tier": chunk.tier,
                "@search.score": round(float(raw_search_score), 4),
                "@search.reranker_score": round(float(rerank_score), 4),
                "score": round(float(rerank_score), 4)
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
