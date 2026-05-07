"""HYDE (Hypothetical Document Embeddings) retrieval strategy."""
from typing import Any

from src.core.document import Document
from src.core.interfaces import BaseEmbedder, BaseLLM, BaseRetriever
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
        llm: BaseLLM | None = None,
        **kwargs: Any
    ):
        super().__init__(embedder=embedder, top_k=top_k, **kwargs)
        if llm is None:
            from src.llms import create_llm, LLMConfig
            llm = create_llm(LLMConfig())
        self._llm = llm

        # Use NaiveRetriever as the underlying retriever
        self._base_retriever = NaiveRetriever(
            embedder=embedder,
            top_k=top_k,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    def _generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = f"""Você é um especialista em nutrição. Escreva um parágrafo curto e informativo
que responderia diretamente à seguinte pergunta. Escreva como se fosse um trecho de um
documento científico ou guia nutricional oficial.

Pergunta: {query}

Resposta (parágrafo informativo):"""

        response = self._llm.generate(prompt)
        return response.strip() if response else ""

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

        # Fallback to direct query embedding if LLM returned nothing
        if hypothetical_doc:
            hyde_embedding = self.embedder.embed_single(hypothetical_doc)
        else:
            if hasattr(self.embedder, 'embed_query'):
                hyde_embedding = self.embedder.embed_query(query)
            else:
                hyde_embedding = self.embedder.embed_single(query)

        # Query using the hypothetical document embedding
        return self._base_retriever.query_by_embedding(hyde_embedding, top_k)

    def clear(self) -> None:
        """Clear all documents."""
        self._base_retriever.clear()
