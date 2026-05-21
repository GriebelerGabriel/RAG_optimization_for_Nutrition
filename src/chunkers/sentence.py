"""Sentence-based chunking strategy."""
import re
from typing import Any

from src.core.document import Document
from src.core.interfaces import BaseChunker

# Pre-compiled regex for sentence splitting (Portuguese/English uppercase letters)
_SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ])'
)


class SentenceChunker(BaseChunker):
    """Chunk text by grouping sentences together."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, sentences_per_chunk: int = 5, **kwargs: Any):
        super().__init__(chunk_size=chunk_size, overlap=overlap, **kwargs)
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = max(1, overlap // 50)  # Convert char overlap to sentence overlap
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = _SENTENCE_SPLIT_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """Split text into sentence-based chunks.
        
        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to each chunk.
            
        Returns:
            List of Document objects.
        """
        if metadata is None:
            metadata = {}
        
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        chunk_index = 0
        i = 0
        
        while i < len(sentences):
            # Get sentences for this chunk
            end_idx = min(i + self.sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            chunk_text = " ".join(chunk_sentences)
            
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "sentence_start": i,
                    "sentence_end": end_idx,
                    "chunker": "sentence",
                }
                chunks.append(Document(content=chunk_text, metadata=chunk_metadata))
                chunk_index += 1
            
            # Move forward, accounting for overlap
            step = max(1, self.sentences_per_chunk - self.overlap_sentences)
            i += step
        
        return chunks
