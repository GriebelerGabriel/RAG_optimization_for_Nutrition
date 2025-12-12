"""Naive retrieval using direct embedding similarity."""
from typing import Any

import chromadb
from chromadb.config import Settings

from src.core.document import Document
from src.core.interfaces import BaseEmbedder, BaseRetriever


class NaiveRetriever(BaseRetriever):
    """Simple retriever using direct query embedding and cosine similarity."""
    
    def __init__(
        self, 
        embedder: BaseEmbedder, 
        top_k: int = 5,
        collection_name: str = "nutrition_docs",
        persist_directory: str = "./chroma_db",
        **kwargs: Any
    ):
        super().__init__(embedder=embedder, top_k=top_k, **kwargs)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Create or get collection
        # Use embedding model name in collection for uniqueness
        model_safe = embedder.model_name.replace("/", "_").replace("-", "_")
        self._collection_name = f"{collection_name}_{model_safe}"
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to ChromaDB collection.
        
        Args:
            documents: Documents to index (must have embeddings).
        """
        if not documents:
            return
        
        # Ensure documents have embeddings
        docs_without_embeddings = [d for d in documents if d.embedding is None]
        if docs_without_embeddings:
            self.embedder.embed_documents(docs_without_embeddings)
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
    
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve documents using direct query embedding.
        
        Args:
            query: Search query.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of relevant documents with scores.
        """
        if top_k is None:
            top_k = self.top_k
        
        # Embed the query
        if hasattr(self.embedder, 'embed_query'):
            query_embedding = self.embedder.embed_query(query)
        else:
            query_embedding = self.embedder.embed_single(query)
        
        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert to Document objects
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances, convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance  # Cosine distance to similarity
                
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=score,
                )
                documents.append(doc)
        
        return documents
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
