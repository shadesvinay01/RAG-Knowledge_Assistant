import os
import sys
from src.azure_search import HybridSearchEngine

def setup_search_index():
    """
    Standalone setup script to provision and initialize the Azure AI Search Index schema
    during initial deployment or setup, without waiting for a Blob trigger.
    """
    print("=" * 70)
    print("🚀 AZURE AI SEARCH INDEX INITIALIZATION & PROVISIONING SCRIPT")
    print("=" * 70)

    engine = HybridSearchEngine()
    
    if engine.use_azure:
        print("[1/2] Connecting to Azure AI Search service endpoint...")
        engine.create_azure_index_schema()
        print("✅ Azure AI Search Index schema provisioned successfully.")
    else:
        print("[Info] Running in Local Hybrid Index mode. Azure cloud credentials not supplied.")

    print("=" * 70)

if __name__ == "__main__":
    setup_search_index()
