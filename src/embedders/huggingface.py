"""HuggingFace sentence-transformers embeddings."""
import logging
import threading
from typing import Any

from sentence_transformers import SentenceTransformer

from src.core.interfaces import BaseEmbedder

logger = logging.getLogger(__name__)

# Module-level model cache: shares loaded models across all HuggingFaceEmbedder instances
# so switching configs doesn't re-load the same ~1GB model.
_model_cache: dict[str, SentenceTransformer] = {}
_cache_lock = threading.Lock()


class HuggingFaceEmbedder(BaseEmbedder):
    """Embedder using HuggingFace sentence-transformers models."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        self._is_e5 = "e5" in model_name.lower()
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        """Thread-safe lazy load the model, cached across instances."""
        if self._model is None:
            with _cache_lock:
                if self.model_name not in _model_cache:
                    logger.info("Loading embedding model: %s (CPU) ...", self.model_name)
                    _model_cache[self.model_name] = SentenceTransformer(self.model_name, device="cpu")
                    logger.info("Embedding model loaded: %s", self.model_name)
                self._model = _model_cache[self.model_name]
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using sentence-transformers.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]

        with self._lock:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a query (with appropriate prefix for E5 models)."""
        if self._is_e5:
            text = f"query: {text}"

        with self._lock:
            embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple queries at once. Much faster than individual calls."""
        if self._is_e5:
            texts = [f"query: {t}" for t in texts]

        with self._lock:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=len(texts))
        return embeddings.tolist()
