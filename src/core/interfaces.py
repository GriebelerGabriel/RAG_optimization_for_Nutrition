"""Abstract base classes for swappable RAG components."""
from abc import ABC, abstractmethod
from typing import Any

from .document import Document

# ---------------------------------------------------------------------------
# Prompt templates — extracted for easy reference (e.g. TCC appendix).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Você é um assistente especialista em nutrição.
Responda APENAS com base no contexto fornecido.
Se o contexto estiver vazio ou não contiver informação relevante para a pergunta, responda: "Não encontrei essa informação nas fontes disponíveis."
NUNCA invente, infira ou use conhecimento externo para responder.
Sempre que houver contexto relevante, cite pelo menos uma fonte no formato [Fonte X].
Seja sempre sucinto e claro, evitando redundâncias e informações irrelevantes."""

USER_PROMPT_TEMPLATE = """\
Contexto:
{context_text}

Pergunta do usuário: {query}

Resposta:"""


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

        user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_text, query=query)

        return self.generate_with_system(system_prompt, user_prompt)

    def generate_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using separate system and user prompts.

        Default implementation concatenates into a single prompt.
        Subclasses (like LiteLLM) override this to use proper message roles.
        """
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        return self.generate(full_prompt)
    
    def _format_context(self, context: list[Document], max_chars: int = 12000) -> str:
        """Format context documents using template or default format.

        Args:
            context: Documents to format.
            max_chars: Approximate character budget to avoid exceeding LLM limits.
        """
        if self.context_template:
            # Custom template - supports {i}, {content}, {source}, {score} placeholders
            formatted_chunks = []
            total = 0
            for i, doc in enumerate(context):
                chunk_text = self.context_template.format(
                    i=i+1,
                    content=doc.content,
                    source=doc.metadata.get("source", "unknown"),
                    score=doc.score or 0.0,
                )
                if total + len(chunk_text) > max_chars:
                    break
                formatted_chunks.append(chunk_text)
                total += len(chunk_text)
            return "\n\n".join(formatted_chunks)

        # Default format
        chunks = []
        total = 0
        for i, doc in enumerate(context):
            text = doc.content[:1000]
            chunk = f"[Fonte {i+1}]: {text}"
            if total + len(chunk) > max_chars:
                break
            chunks.append(chunk)
            total += len(chunk)
        return "\n\n".join(chunks)
    
    def _default_system_prompt(self) -> str:
        return SYSTEM_PROMPT
