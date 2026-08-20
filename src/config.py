import os
from pydantic import BaseModel

class Config(BaseModel):
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-azure-openai.openai.azure.com/")
    AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

    # Azure AI Search Configuration
    AZURE_SEARCH_ENDPOINT: str = os.getenv("AZURE_SEARCH_ENDPOINT", "https://your-azure-search.search.windows.net")
    AZURE_SEARCH_KEY: str = os.getenv("AZURE_SEARCH_KEY", "")
    AZURE_SEARCH_INDEX: str = os.getenv("AZURE_SEARCH_INDEX", "enterprise-knowledge-index")

    # Local / Offline Fallback Configuration
    USE_LOCAL_FALLBACK: bool = True
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # RAG Hyperparameters & Thresholds
    BASELINE_TOP_K: int = 3
    IMPROVED_TOP_K: int = 5
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 80
    GROUNDEDNESS_THRESHOLD: float = 0.45
    CACHE_SIMILARITY_THRESHOLD: float = 0.92

config = Config()
