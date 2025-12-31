"""Abstract base classes for swappable RAG components."""
from abc import ABC, abstractmethod
from typing import Any

from .document import Document


class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    def __init__(self, **kwargs: Any):
        self.config = kwargs
    
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Document]:
        """Split text into chunks.
        
        Args:
            text: The text to chunk.
            metadata: Optional metadata to attach to each chunk.
            
        Returns:
            List of Document objects.
        """
        ...
    
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk multiple documents."""
        chunks = []
        for doc in documents:
            chunks.extend(self.chunk(doc.content, doc.metadata))
        return chunks


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, **kwargs: Any):
        self.config = kwargs
        self.model_name: str = kwargs.get("model_name", "unknown")
    
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        ...
    
    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]
    
    def embed_documents(self, documents: list[Document]) -> list[Document]:
        """Add embeddings to documents."""
        texts = [doc.content for doc in documents]
        embeddings = self.embed(texts)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        return documents


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies."""
    
    def __init__(self, embedder: BaseEmbedder, **kwargs: Any):
        self.embedder = embedder
        self.config = kwargs
        self.top_k: int = kwargs.get("top_k", 5)
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The search query.
            top_k: Number of documents to retrieve (overrides default).
            
        Returns:
            List of relevant Documents with scores.
        """
        ...
    
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the retriever's index.
        
        Args:
            documents: Documents to index.
        """
        ...


class BaseReranker(ABC):
    """Abstract base class for reranking models."""
    
    def __init__(self, **kwargs: Any):
        self.config = kwargs
        self.top_k: int = kwargs.get("top_k", 5)
    
    @abstractmethod
    def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        """Rerank documents based on relevance to query.
        
        Args:
            query: The search query.
            documents: Documents to rerank.
            top_k: Number of documents to return after reranking.
            
        Returns:
            Reranked list of Documents with updated scores.
        """
        ...


class BaseLLM(ABC):
    """Abstract base class for language models."""
    
    def __init__(self, **kwargs: Any):
        self.config = kwargs
        self.model_name: str = kwargs.get("model_name", "unknown")
        self.temperature: float = kwargs.get("temperature", 0.1)
        self.system_prompt: str | None = kwargs.get("system_prompt")
        self.context_template: str | None = kwargs.get("context_template")
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            Generated text.
        """
        ...
    
    def generate_with_context(self, query: str, context: list[Document], system_prompt: str | None = None) -> str:
        """Generate response using retrieved context.
        
        Args:
            query: User's question.
            context: Retrieved documents.
            system_prompt: Optional system prompt override.
            
        Returns:
            Generated response.
        """
        # Priority: explicit param > instance config > default
        if system_prompt is None:
            system_prompt = self.system_prompt or self._default_system_prompt()
        
        # Format context using template or default
        context_text = self._format_context(context)
        
        full_prompt = f"""{system_prompt}

Contexto:
{context_text}

Pergunta do usuário: {query}

Resposta:"""
        
        return self.generate(full_prompt)
    
    def _format_context(self, context: list[Document]) -> str:
        """Format context documents using template or default format."""
        if self.context_template:
            # Custom template - supports {i}, {content}, {source}, {score} placeholders
            formatted_chunks = []
            for i, doc in enumerate(context):
                chunk_text = self.context_template.format(
                    i=i+1,
                    content=doc.content,
                    source=doc.metadata.get("source", "unknown"),
                    score=doc.score or 0.0,
                )
                formatted_chunks.append(chunk_text)
            return "\n\n".join(formatted_chunks)
        
        # Default format
        return "\n\n".join(
            [
                f"[Fonte {i+1}]: {doc.content[:1000]}" 
                for i, doc in enumerate(context)
            ]
        )
    
    def _default_system_prompt(self) -> str:
        return """Você é um nutricionista especializado em educação alimentar.
Responda APENAS com base no contexto fornecido.
Se a informação não estiver no contexto, diga "Não encontrei essa informação nas fontes disponíveis."

Instruções:
- Seja factual e cite as fontes quando possível
- Use linguagem acessível ao público geral
- Não faça recomendações médicas individualizadas"""
