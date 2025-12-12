"""HYDE (Hypothetical Document Embeddings) retrieval strategy."""
import os
from typing import Any

from openai import OpenAI

from src.core.document import Document
from src.core.interfaces import BaseEmbedder, BaseRetriever
from .naive import NaiveRetriever


class HYDERetriever(BaseRetriever):
    """Retriever using Hypothetical Document Embeddings.
    
    Instead of embedding the query directly, HYDE first generates a
    hypothetical answer to the query, then embeds that answer for retrieval.
    This often improves retrieval for complex questions.
    """
    
    def __init__(
        self, 
        embedder: BaseEmbedder, 
        top_k: int = 5,
        collection_name: str = "nutrition_docs",
        persist_directory: str = "./chroma_db",
        llm_model: str = "gpt-4o-mini",
        **kwargs: Any
    ):
        super().__init__(embedder=embedder, top_k=top_k, **kwargs)
        self.llm_model = llm_model
        
        # Use NaiveRetriever as the underlying retriever
        self._base_retriever = NaiveRetriever(
            embedder=embedder,
            top_k=top_k,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        
        self._client: OpenAI | None = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy load OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required for HYDE retriever")
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = f"""Você é um especialista em nutrição. Escreva um parágrafo curto e informativo 
que responderia diretamente à seguinte pergunta. Escreva como se fosse um trecho de um 
documento científico ou guia nutricional oficial.

Pergunta: {query}

Resposta (parágrafo informativo):"""
        
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        
        return response.choices[0].message.content or ""
    
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the underlying retriever."""
        self._base_retriever.add_documents(documents)
    
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve using hypothetical document embedding.
        
        Args:
            query: Search query.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of relevant documents with scores.
        """
        if top_k is None:
            top_k = self.top_k
        
        # Generate hypothetical document
        hypothetical_doc = self._generate_hypothetical_document(query)
        
        # Embed the hypothetical document instead of the query
        if hasattr(self.embedder, 'embed_query'):
            # For E5 models, use passage prefix for hypothetical doc
            hyde_embedding = self.embedder.embed_single(hypothetical_doc)
        else:
            hyde_embedding = self.embedder.embed_single(hypothetical_doc)
        
        # Query using the hypothetical document embedding
        results = self._base_retriever._collection.query(
            query_embeddings=[hyde_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        # Convert to Document objects
        documents = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                score = 1 - distance
                
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=score,
                )
                documents.append(doc)
        
        return documents
    
    def clear(self) -> None:
        """Clear all documents."""
        self._base_retriever.clear()
