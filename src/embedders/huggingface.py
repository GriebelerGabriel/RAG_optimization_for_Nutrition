"""HuggingFace sentence-transformers embeddings."""
from typing import Any

from sentence_transformers import SentenceTransformer

from src.core.interfaces import BaseEmbedder


class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder using HuggingFace sentence-transformers models."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        self._is_e5 = "e5" in model_name.lower()
        self._model: SentenceTransformer | None = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using sentence-transformers.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        # E5 models require "query: " or "passage: " prefix
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a query (with appropriate prefix for E5 models)."""
        if self._is_e5:
            text = f"query: {text}"
        
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()
