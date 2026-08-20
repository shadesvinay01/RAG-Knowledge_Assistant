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
    AZURE_SEARCH_SEMANTIC_CONFIG: str = "enterprise-semantic-config"

    # Application Insights Observability
    APPLICATIONINSIGHTS_CONNECTION_STRING: str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")

    # Pricing Parameters (for Cost Estimation per Query)
    COST_PER_1K_INPUT_TOKENS: float = 0.005      # gpt-4o input cost ($0.005 / 1k tokens)
    COST_PER_1K_OUTPUT_TOKENS: float = 0.015     # gpt-4o output cost ($0.015 / 1k tokens)
    COST_PER_1K_EMBEDDING_TOKENS: float = 0.00013 # text-embedding-3-large ($0.00013 / 1k tokens)

    # Local / Offline Fallback Configuration
    USE_LOCAL_FALLBACK: bool = True
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # RAG Hyperparameters & Thresholds
    BASELINE_TOP_K: int = 3
    IMPROVED_TOP_K: int = 5
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 80
    GROUNDEDNESS_THRESHOLD: float = 0.40
    CACHE_SIMILARITY_THRESHOLD: float = 0.92

config = Config()
