"""Retrieval strategies for document search."""
from typing import TYPE_CHECKING

from src.core.interfaces import BaseEmbedder, BaseRetriever

if TYPE_CHECKING:
    from src.core.config import RetrieverConfig, Config

from .naive import NaiveRetriever
from .hyde import HYDERetriever
from .hybrid import HybridRetriever

__all__ = ["NaiveRetriever", "HYDERetriever", "HybridRetriever", "create_retriever"]

RETRIEVER_REGISTRY: dict[str, type[BaseRetriever]] = {
    "naive": NaiveRetriever,
    "hyde": HYDERetriever,
    "hybrid": HybridRetriever,
}


def create_retriever(
    config: "RetrieverConfig", 
    embedder: BaseEmbedder,
    full_config: "Config | None" = None,
) -> BaseRetriever:
    """Factory function to create retriever from config.
    
    Args:
        config: RetrieverConfig with retriever settings.
        embedder: Embedder instance to use.
        full_config: Optional full Config for generating unique collection names
                     based on config hash (prevents cross-run contamination).
    """
    retriever_class = RETRIEVER_REGISTRY.get(config.type)
    if retriever_class is None:
        raise ValueError(f"Unknown retriever type: {config.type}. Available: {list(RETRIEVER_REGISTRY.keys())}")
    
    # Use config-hash-based collection name if full config provided
    if full_config is not None:
        collection_name = full_config.get_unique_collection_name()
    else:
        collection_name = config.collection_name
    
    return retriever_class(
        embedder=embedder,
        top_k=config.top_k,
        collection_name=collection_name,
    )
