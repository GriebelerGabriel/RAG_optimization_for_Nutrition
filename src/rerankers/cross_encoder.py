"""Cross-encoder reranking model."""
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from src.core.document import Document
from src.core.interfaces import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder models for more accurate relevance scoring."""
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        **kwargs: Any
    ):
        super().__init__(top_k=top_k, **kwargs)
        self.model_name = model_name
        self._model: CrossEncoder | None = None
    
    @property
    def model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model
    
    def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        """Rerank documents using cross-encoder.
        
        Args:
            query: Search query.
            documents: Documents to rerank.
            top_k: Number of documents to return.
            
        Returns:
            Reranked documents with updated scores.
        """
        if top_k is None:
            top_k = self.top_k
        
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc.content] for doc in documents]
        
        # Get cross-encoder scores (raw logits, typically [-10, +10])
        scores = self.model.predict(pairs)

        # Normalize to [0, 1] via sigmoid so scores are comparable with
        # cosine similarity thresholds used downstream.
        scores = 1 / (1 + np.exp(-np.array(scores)))

        # Create scored copies to avoid mutating input
        from src.core.document import Document
        scored_docs = [
            Document(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                embedding=doc.embedding,
                score=float(score),
            )
            for doc, score in zip(documents, scores)
        ]

        # Sort by score descending
        reranked = sorted(scored_docs, key=lambda x: x.score or 0, reverse=True)
        
        return reranked[:top_k]
