import numpy as np
from typing import List
from src.config import config

class EmbeddingEngine:
    """
    Embedding Engine supporting Azure OpenAI text-embedding-3-large with explicit 1536-dimension specification
    and local sentence-transformers fallback.
    """
    def __init__(self):
        self.use_azure = bool(config.AZURE_OPENAI_KEY and "your-azure" not in config.AZURE_OPENAI_ENDPOINT)
        self.local_model = None

        if self.use_azure:
            try:
                from openai import AzureOpenAI
                self.client = AzureOpenAI(
                    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                    api_key=config.AZURE_OPENAI_KEY,
                    api_version=config.AZURE_OPENAI_API_VERSION
                )
            except Exception:
                self.use_azure = False

        if not self.use_azure:
            try:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            except Exception:
                self.local_model = None

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.use_azure:
            try:
                # Explicitly request 1536 dimensions for text-embedding-3-large to match Azure AI Search schema
                res = self.client.embeddings.create(
                    input=texts,
                    model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                    dimensions=1536
                )
                return [d.embedding for d in res.data]
            except Exception as e:
                print(f"[Warning] Azure embedding failed ({e}), falling back to local model.")

        if self.local_model:
            embeddings = self.local_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        # Deterministic lightweight hash vectorizer fallback (384 dimensions)
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        res = self.embed_texts([text])
        return res[0] if res else []

    def _hash_vector(self, text: str, dim: int = 384) -> List[float]:
        vec = np.zeros(dim)
        words = text.lower().split()
        for i, word in enumerate(words):
            idx = abs(hash(word)) % dim
            vec[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
