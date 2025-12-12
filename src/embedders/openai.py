"""OpenAI embeddings."""
import os
from typing import Any

from openai import OpenAI

from src.core.interfaces import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI's embedding API."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        self._client: OpenAI | None = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy load the OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        # OpenAI API has a limit on batch size
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
