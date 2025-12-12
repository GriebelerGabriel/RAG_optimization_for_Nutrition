"""Document model for RAG system."""
from dataclasses import dataclass, field
from typing import Any


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
            import hashlib
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(id={self.id}, score={self.score}, content='{preview}')"
