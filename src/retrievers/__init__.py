"""Retrieval strategies for document search."""
from typing import TYPE_CHECKING

from src.core.interfaces import BaseEmbedder, BaseRetriever

if TYPE_CHECKING:
    from src.core.config import RetrieverConfig

from .naive import NaiveRetriever
from .hyde import HYDERetriever
from .hybrid import HybridRetriever

__all__ = ["NaiveRetriever", "HYDERetriever", "HybridRetriever", "create_retriever"]

RETRIEVER_REGISTRY: dict[str, type[BaseRetriever]] = {
    "naive": NaiveRetriever,
    "hyde": HYDERetriever,
    "hybrid": HybridRetriever,
}


def create_retriever(config: "RetrieverConfig", embedder: BaseEmbedder) -> BaseRetriever:
    """Factory function to create retriever from config."""
    retriever_class = RETRIEVER_REGISTRY.get(config.type)
    if retriever_class is None:
        raise ValueError(f"Unknown retriever type: {config.type}. Available: {list(RETRIEVER_REGISTRY.keys())}")
    
    return retriever_class(
        embedder=embedder,
        top_k=config.top_k,
        collection_name=config.collection_name,
    )
