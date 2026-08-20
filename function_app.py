import logging
import azure.functions as func
from src.ingestion import DocumentIngestionPipeline
from src.embeddings import EmbeddingEngine
from src.azure_search import HybridSearchEngine

app = func.FunctionApp()

@app.blob_trigger(
    arg_name="myblob",
    path="knowledge-base-hot/{name}",
    connection="AZURE_STORAGE_CONNECTION_STRING"
)
def blob_ingestion_trigger(myblob: func.InputStream):
    """
    Azure Function Event Grid Blob Trigger:
    Triggered automatically whenever a new enterprise document (PDF/DOCX/MD) is uploaded to Azure Blob Storage.
    Processes raw binary bytes via Azure Document Intelligence OCR, generates 1536-dim embeddings via Azure OpenAI,
    and indexes document metadata into Azure AI Search.
    """
    logging.info(f"⚡ Azure Function Blob Trigger processing blob: {myblob.name} ({myblob.length} bytes)")

    try:
        blob_bytes = myblob.read() # Read raw binary bytes
        filename = myblob.name.split("/")[-1]

        ingestion_pipeline = DocumentIngestionPipeline()
        chunks = ingestion_pipeline.parse_blob_file(blob_bytes, filename)
        logging.info(f"   Generated {len(chunks)} chunks from {filename}.")

        embedding_engine = EmbeddingEngine()
        chunk_texts = [c.content for c in chunks]
        vectors = embedding_engine.embed_texts(chunk_texts)

        search_engine = HybridSearchEngine()
        search_engine.create_azure_index_schema()
        search_engine.index_chunks(chunks, vectors)

        logging.info(f"✅ Successfully indexed {len(chunks)} chunks from {filename} into Azure AI Search.")
    except Exception as e:
        logging.error(f"❌ Error during Azure Function ingestion: {e}")
