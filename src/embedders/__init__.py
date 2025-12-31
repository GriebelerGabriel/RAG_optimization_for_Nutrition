"""Embedding models for vector representations."""
from typing import TYPE_CHECKING

from src.core.interfaces import BaseEmbedder

if TYPE_CHECKING:
    from src.core.config import EmbedderConfig

from .huggingface import HuggingFaceEmbedder
from .openai import OpenAIEmbedder

__all__ = ["HuggingFaceEmbedder", "OpenAIEmbedder", "create_embedder"]

EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {
    "e5": HuggingFaceEmbedder,
    "mpnet": HuggingFaceEmbedder,
    "bge": HuggingFaceEmbedder,
    "huggingface": HuggingFaceEmbedder,
    "openai": OpenAIEmbedder,
}

# Default model names for each type
DEFAULT_MODELS: dict[str, str] = {
    "e5": "intfloat/multilingual-e5-base",
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge": "BAAI/bge-m3",
}


def create_embedder(config: "EmbedderConfig") -> BaseEmbedder:
    """Factory function to create embedder from config."""
    embedder_class = EMBEDDER_REGISTRY.get(config.type)
    if embedder_class is None:
        raise ValueError(f"Unknown embedder type: {config.type}. Available: {list(EMBEDDER_REGISTRY.keys())}")
    
    # Use default model if not specified
    model_name = config.model_name
    if model_name == "" or model_name is None:
        model_name = DEFAULT_MODELS.get(config.type, config.model_name)
    
    return embedder_class(model_name=model_name)
