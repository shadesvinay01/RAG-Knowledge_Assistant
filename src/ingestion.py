import os
import io
import re
from typing import List, Dict, Any, Optional
from src.chunking import HierarchicalChunker, Chunk

class AzureBlobStorageClient:
    """
    Azure Blob Storage SDK Integration for document ingestion from hot container storage.
    """
    def __init__(self):
        self.conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER", "knowledge-base-hot")
        self.blob_service_client = None

        if self.conn_str:
            try:
                from azure.storage.blob import BlobServiceClient
                self.blob_service_client = BlobServiceClient.from_connection_string(self.conn_str)
            except Exception as e:
                print(f"[Warning] Azure Blob Storage client skipped ({e}). Falling back to local directory ingestion.")

    def download_blobs(self) -> List[Dict[str, Any]]:
        documents = []
        if self.blob_service_client:
            try:
                container_client = self.blob_service_client.get_container_client(self.container_name)
                for blob in container_client.list_blobs():
                    blob_client = container_client.get_blob_client(blob.name)
                    blob_bytes = blob_client.download_blob().readall()
                    documents.append({"filename": blob.name, "bytes": blob_bytes})
                return documents
            except Exception as e:
                print(f"[Warning] Blob download failed: {e}")
        return []


class AzureDocumentIntelligenceParser:
    """
    Azure AI Document Intelligence SDK Integration for binary PDF/DOCX OCR, layout parsing, and table extraction.
    Safe handling: Binary documents are parsed via Document Intelligence layout or PyPDF parser.
    """
    def __init__(self):
        self.endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
        self.key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
        self.client = None

        if self.endpoint and self.key:
            try:
                from azure.ai.documentintelligence import DocumentIntelligenceClient
                from azure.core.credentials import AzureKeyCredential
                self.client = DocumentIntelligenceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key))
            except Exception as e:
                print(f"[Warning] Azure Document Intelligence client skipped ({e}). Using structural parser.")

    def parse_blob(self, blob_bytes: bytes, filename: str) -> str:
        """
        Safely parses binary PDF/DOCX or text/MD blobs into structured text content.
        Binary PDF files route to Document Intelligence layout or PyPDF parser.
        """
        is_binary = filename.endswith(".pdf") or filename.endswith(".docx") or filename.endswith(".bin")

        if is_binary:
            if self.client:
                try:
                    poller = self.client.begin_analyze_document(
                        model_id="prebuilt-layout",
                        analyze_request=blob_bytes,
                        content_type="application/octet-stream"
                    )
                    result = poller.result()
                    return result.content
                except Exception as e:
                    print(f"[Warning] Azure Document Intelligence PDF analysis failed: {e}")

            # PyPDF binary parser fallback
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(blob_bytes))
                text_pages = [page.extract_text() for page in reader.pages if page.extract_text()]
                if text_pages:
                    return "\n\n".join(text_pages)
            except Exception:
                pass

            print(f"[Ingestion Log] Could not extract text from binary document '{filename}'. File flagged for manual handling/retry.")
            return f"# Document: {filename}\n[Binary document processing failed; document requires manual handling/retry]."


        # UTF-8 text/Markdown parsing
        try:
            return blob_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return blob_bytes.decode("latin-1", errors="ignore")


class DocumentIngestionPipeline:
    """
    Production Document Ingestion Pipeline connecting Azure Blob Storage,
    Azure Document Intelligence parsing, and Hierarchical Chunking.
    """
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.blob_client = AzureBlobStorageClient()
        self.doc_parser = AzureDocumentIntelligenceParser()
        self.chunker = HierarchicalChunker(chunk_size=chunk_size, overlap=overlap)

    def parse_blob_file(self, blob_bytes: bytes, filename: str) -> List[Chunk]:
        parsed_content = self.doc_parser.parse_blob(blob_bytes, filename)
        chunks = self.chunker.chunk_document(parsed_content, filename)
        return chunks

    def ingest_all(self, local_data_dir: str = "data/knowledge_base") -> List[Chunk]:
        all_chunks = []
        blob_docs = self.blob_client.download_blobs()

        if blob_docs:
            print(f"✅ Downloaded {len(blob_docs)} document blobs from Azure Blob Storage container.")
            for doc in blob_docs:
                chunks = self.parse_blob_file(doc["bytes"], doc["filename"])
                all_chunks.extend(chunks)
            return all_chunks

        if os.path.exists(local_data_dir):
            for filename in sorted(os.listdir(local_data_dir)):
                if filename.endswith(".md") or filename.endswith(".txt") or filename.endswith(".pdf"):
                    filepath = os.path.join(local_data_dir, filename)
                    with open(filepath, "rb") as f:
                        file_bytes = f.read()
                    chunks = self.parse_blob_file(file_bytes, filename)
                    all_chunks.extend(chunks)

        return all_chunks
