"""Document model for RAG system."""
from dataclasses import dataclass, field
from typing import Any
import hashlib

@dataclass
class Document:
    """Represents a document chunk with content and metadata."""
    
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    score: float | None = None
    embedding: list[float] | None = None
    
    def __post_init__(self):
        if self.id is None:
            source = self.metadata.get("source") or self.metadata.get("filename") or ""
            chunker = self.metadata.get("chunker") or ""
            chunk_index = self.metadata.get("chunk_index")
            chunk_start = self.metadata.get("chunk_start")
            chunk_end = self.metadata.get("chunk_end")
            sentence_start = self.metadata.get("sentence_start")
            sentence_end = self.metadata.get("sentence_end")

            identity_parts = [
                f"source={source}",
                f"chunker={chunker}",
                f"chunk_index={chunk_index}",
                f"chunk_start={chunk_start}",
                f"chunk_end={chunk_end}",
                f"sentence_start={sentence_start}",
                f"sentence_end={sentence_end}",
                f"content={self.content}",
            ]
            identity = "|".join(identity_parts)
            self.id = hashlib.md5(identity.encode("utf-8")).hexdigest()[:12]
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id}, score={self.score}, content='{preview}')"
