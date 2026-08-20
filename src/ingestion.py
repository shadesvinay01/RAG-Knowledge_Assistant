import os
import re
from typing import List, Dict, Any
from src.chunking import HierarchicalChunker, Chunk

class DocumentIngestionPipeline:
    """
    Production Document Ingestion Pipeline simulating Azure Blob Storage ingestion & 
    Azure AI Document Intelligence parsing. Processes Markdown/PDF text, extracts metadata
    (effective_date, version, status, department, tier), section headers, and generates chunks.
    """
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunker = HierarchicalChunker(chunk_size=chunk_size, overlap=overlap)

    def parse_document(self, content: str, filename: str) -> List[Chunk]:
        """
        Parses document content, identifies structure/headers/tables/metadata,
        and generates structured chunks with document-level ACL and version tags.
        """
        chunks = self.chunker.chunk_document(content, filename)
        return chunks

    def ingest_directory(self, data_dir: str) -> List[Chunk]:
        """
        Ingests all documents in a directory (simulating Azure Blob Storage container ingestion).
        """
        all_chunks = []
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Knowledge base directory '{data_dir}' not found.")

        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith(".md") or filename.endswith(".txt"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                chunks = self.parse_document(text, filename)
                all_chunks.extend(chunks)

        return all_chunks
