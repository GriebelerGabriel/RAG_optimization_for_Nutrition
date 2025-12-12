"""No-op reranker (passthrough)."""
from typing import Any

from src.core.document import Document
from src.core.interfaces import BaseReranker


class NoReranker(BaseReranker):
    """Passthrough reranker that returns documents unchanged."""
    
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
    
    def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        """Return documents unchanged.
        
        Args:
            query: Search query (unused).
            documents: Documents to return.
            top_k: Number of documents to return.
            
        Returns:
            Original documents (possibly truncated to top_k).
        """
        if top_k is None:
            top_k = self.top_k
        
        return documents[:top_k]
