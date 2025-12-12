"""RAG Pipeline orchestrator."""
from pathlib import Path
from typing import Any

from .config import Config, load_config
from .document import Document
from .interfaces import BaseChunker, BaseEmbedder, BaseLLM, BaseReranker, BaseRetriever


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""
    
    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        reranker: BaseReranker | None,
        llm: BaseLLM,
        config: Config | None = None,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.config = config
    
    @classmethod
    def from_config(cls, config: Config | str | Path) -> "RAGPipeline":
        """Create pipeline from configuration.
        
        Args:
            config: Config object or path to YAML config file.
            
        Returns:
            Configured RAGPipeline instance.
        """
        if isinstance(config, (str, Path)):
            config = load_config(config)
        
        # Import component factories
        from src.chunkers import create_chunker
        from src.embedders import create_embedder
        from src.retrievers import create_retriever
        from src.rerankers import create_reranker
        from src.llms import create_llm
        
        # Create components
        chunker = create_chunker(config.chunker)
        embedder = create_embedder(config.embedding)
        retriever = create_retriever(config.retriever, embedder, full_config=config)
        reranker = create_reranker(config.reranker)
        llm = create_llm(config.llm)
        
        return cls(
            chunker=chunker,
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            config=config,
        )
    
    def ingest(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """Ingest text into the pipeline.
        
        Args:
            text: Text content to ingest.
            metadata: Optional metadata for the document.
            
        Returns:
            List of chunked and indexed documents.
        """
        # Chunk the text
        chunks = self.chunker.chunk(text, metadata)
        
        # Generate embeddings
        chunks = self.embedder.embed_documents(chunks)
        
        # Add to retriever index
        self.retriever.add_documents(chunks)
        
        return chunks
    
    def ingest_documents(self, documents: list[Document]) -> list[Document]:
        """Ingest multiple documents into the pipeline."""
        all_chunks = []
        for doc in documents:
            chunks = self.ingest(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks
    
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Search query.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of relevant documents.
        """
        # Get initial retrieval results
        docs = self.retriever.retrieve(query, top_k)
        
        # Apply reranking if configured
        if self.reranker is not None:
            docs = self.reranker.rerank(query, docs, top_k)
        
        return docs
    
    def generate(self, query: str, context: list[Document]) -> str:
        """Generate response using retrieved context.
        
        Args:
            query: User's question.
            context: Retrieved documents.
            
        Returns:
            Generated response.
        """
        return self.llm.generate_with_context(query, context)
    
    def query(self, query: str, top_k: int | None = None) -> tuple[str, list[Document]]:
        """Full RAG query: retrieve + generate.
        
        Args:
            query: User's question.
            top_k: Number of documents to retrieve.
            
        Returns:
            Tuple of (generated response, retrieved documents).
        """
        # Retrieve relevant documents
        docs = self.retrieve(query, top_k)
        
        # Generate response
        response = self.generate(query, docs)
        
        return response, docs
    
    def __repr__(self) -> str:
        return (
            f"RAGPipeline(\n"
            f"  chunker={self.chunker.__class__.__name__},\n"
            f"  embedder={self.embedder.__class__.__name__},\n"
            f"  retriever={self.retriever.__class__.__name__},\n"
            f"  reranker={self.reranker.__class__.__name__ if self.reranker else 'None'},\n"
            f"  llm={self.llm.__class__.__name__}\n"
            f")"
        )
