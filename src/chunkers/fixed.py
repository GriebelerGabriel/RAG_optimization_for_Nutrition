"""Fixed-size chunking strategy."""
from typing import Any

from src.core.document import Document
from src.core.interfaces import BaseChunker


class FixedChunker(BaseChunker):
    """Chunk text into fixed-size pieces with overlap."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, **kwargs: Any):
        super().__init__(chunk_size=chunk_size, overlap=overlap, **kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """Split text into fixed-size chunks.
        
        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to each chunk.
            
        Returns:
            List of Document objects.
        """
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at a natural boundary (space, newline)
            if end < len(text):
                # Look for last space or newline in the chunk
                last_break = max(
                    chunk_text.rfind(" "),
                    chunk_text.rfind("\n"),
                )
                if last_break > self.chunk_size // 2:  # Only if we found a reasonable break point
                    chunk_text = chunk_text[:last_break]
                    end = start + last_break
            
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunker": "fixed",
                }
                chunks.append(Document(content=chunk_text, metadata=chunk_metadata))
                chunk_index += 1
            
            # Move start position, accounting for overlap
            start = end - self.overlap
            if start <= chunks[-1].metadata["chunk_start"] if chunks else 0:
                start = end  # Prevent infinite loop
        
        return chunks
