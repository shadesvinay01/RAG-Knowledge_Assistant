import re
from typing import List, Dict, Any

class Chunk:
    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        doc_name: str,
        content: str,
        header: str,
        effective_date: str = "2024-01-01",
        version: str = "1.0",
        department: str = "All",
        tier: str = "All",
        parent_content: str = "",
        chunk_index: int = 0
    ):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.content = content
        self.header = header
        self.effective_date = effective_date
        self.version = version
        self.department = department
        self.tier = tier
        self.parent_content = parent_content
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "content": self.content,
            "header": self.header,
            "effective_date": self.effective_date,
            "version": self.version,
            "department": self.department,
            "tier": self.tier,
            "parent_content": self.parent_content,
            "chunk_index": self.chunk_index
        }

class HierarchicalChunker:
    """
    Advanced semantic & hierarchical chunker.
    Parses document headers, extracts metadata (effective date, version, department, tier),
    and creates chunks with parent-child linkage to prevent 'Wrong Chunk' failure mode.
    """
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_metadata(self, text: str, filename: str) -> Dict[str, str]:
        meta = {
            "doc_id": filename.replace(".md", "").replace(".pdf", ""),
            "doc_name": filename,
            "effective_date": "2025-01-01",
            "version": "2025.1",
            "department": "All",
            "tier": "All"
        }
        
        eff_match = re.search(r"\*\*Effective Date:\*\*\s*([^\n\r]+)", text)
        if eff_match:
            meta["effective_date"] = eff_match.group(1).strip()
            
        doc_id_match = re.search(r"\*\*Document ID:\*\*\s*([^\n\r]+)", text)
        if doc_id_match:
            meta["doc_id"] = doc_id_match.group(1).strip()
            
        dept_match = re.search(r"\*\*Department:\*\*\s*([^\n\r]+)", text)
        if dept_match:
            meta["department"] = dept_match.group(1).strip()

        tier_match = re.search(r"\*\*Plan Tier:\*\*\s*([^\n\r]+)", text)
        if tier_match:
            meta["tier"] = tier_match.group(1).strip()

        if "2024" in filename or "2024" in text:
            meta["version"] = "2024"
        elif "2026" in filename or "2026" in text:
            meta["version"] = "2026"

        return meta

    def chunk_document(self, text: str, filename: str) -> List[Chunk]:
        metadata = self.extract_metadata(text, filename)
        sections = re.split(r'\n(?=##?\s+)', text)
        chunks = []
        global_chunk_idx = 0

        for section in sections:
            section_clean = section.strip()
            if not section_clean:
                continue

            header_match = re.match(r'^(##?\s+[^\n\r]+)', section_clean)
            header = header_match.group(1).replace('#', '').strip() if header_match else "Overview"
            
            # Words tokenization split with overlap
            words = section_clean.split()
            if len(words) <= self.chunk_size:
                c = Chunk(
                    chunk_id=f"{metadata['doc_id']}_c{global_chunk_idx}",
                    doc_id=metadata["doc_id"],
                    doc_name=metadata["doc_name"],
                    content=section_clean,
                    header=header,
                    effective_date=metadata["effective_date"],
                    version=metadata["version"],
                    department=metadata["department"],
                    tier=metadata["tier"],
                    parent_content=section_clean,
                    chunk_index=global_chunk_idx
                )
                chunks.append(c)
                global_chunk_idx += 1
            else:
                step = self.chunk_size - self.overlap
                for i in range(0, len(words), step):
                    sub_words = words[i:i + self.chunk_size]
                    sub_content = " ".join(sub_words)
                    c = Chunk(
                        chunk_id=f"{metadata['doc_id']}_c{global_chunk_idx}",
                        doc_id=metadata["doc_id"],
                        doc_name=metadata["doc_name"],
                        content=f"[{header}] {sub_content}",
                        header=header,
                        effective_date=metadata["effective_date"],
                        version=metadata["version"],
                        department=metadata["department"],
                        tier=metadata["tier"],
                        parent_content=section_clean, # Full section preserved as parent context
                        chunk_index=global_chunk_idx
                    )
                    chunks.append(c)
                    global_chunk_idx += 1

        return chunks
