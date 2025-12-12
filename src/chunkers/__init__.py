"""Chunking strategies for document processing."""
from typing import TYPE_CHECKING

from src.core.interfaces import BaseChunker

if TYPE_CHECKING:
    from src.core.config import ChunkerConfig

from .fixed import FixedChunker
from .recursive import RecursiveChunker
from .sentence import SentenceChunker

__all__ = ["FixedChunker", "RecursiveChunker", "SentenceChunker", "create_chunker"]

CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {
    "fixed": FixedChunker,
    "recursive": RecursiveChunker,
    "sentence": SentenceChunker,
}


def create_chunker(config: "ChunkerConfig") -> BaseChunker:
    """Factory function to create chunker from config."""
    chunker_class = CHUNKER_REGISTRY.get(config.type)
    if chunker_class is None:
        raise ValueError(f"Unknown chunker type: {config.type}. Available: {list(CHUNKER_REGISTRY.keys())}")
    
    return chunker_class(
        chunk_size=config.chunk_size,
        overlap=config.overlap,
    )
