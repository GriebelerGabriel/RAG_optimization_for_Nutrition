"""Reranking models for improving retrieval precision."""
from typing import TYPE_CHECKING

from src.core.interfaces import BaseReranker

if TYPE_CHECKING:
    from src.core.config import RerankerConfig

from .none import NoReranker
from .cross_encoder import CrossEncoderReranker

__all__ = ["NoReranker", "CrossEncoderReranker", "create_reranker"]

RERANKER_REGISTRY: dict[str, type[BaseReranker]] = {
    "none": NoReranker,
    "cross_encoder": CrossEncoderReranker,
    "bge": CrossEncoderReranker,
}

DEFAULT_MODELS: dict[str, str] = {
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge": "BAAI/bge-reranker-base",
}


def create_reranker(config: "RerankerConfig") -> BaseReranker | None:
    """Factory function to create reranker from config."""
    if config.type == "none":
        return None
    
    reranker_class = RERANKER_REGISTRY.get(config.type)
    if reranker_class is None:
        raise ValueError(f"Unknown reranker type: {config.type}. Available: {list(RERANKER_REGISTRY.keys())}")
    
    model_name = config.model_name
    if not model_name or model_name == "":
        model_name = DEFAULT_MODELS.get(config.type, "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    return reranker_class(
        model_name=model_name,
        top_k=config.top_k,
    )
