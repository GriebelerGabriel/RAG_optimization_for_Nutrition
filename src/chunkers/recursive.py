"""Recursive character text splitting strategy."""
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.core.document import Document
from src.core.interfaces import BaseChunker


class RecursiveChunker(BaseChunker):
    """Chunk text using LangChain's RecursiveCharacterTextSplitter.
    
    This splitter tries to split on paragraphs first, then sentences,
    then words, preserving semantic coherence.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50, **kwargs: Any):
        super().__init__(chunk_size=chunk_size, overlap=overlap, **kwargs)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """Split text using recursive character splitting.
        
        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to each chunk.
            
        Returns:
            List of Document objects.
        """
        if metadata is None:
            metadata = {}
        
        # Use LangChain splitter
        texts = self.splitter.split_text(text)
        
        chunks = []
        for i, chunk_text in enumerate(texts):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "chunker": "recursive",
                }
                chunks.append(Document(content=chunk_text, metadata=chunk_metadata))
        
        return chunks
