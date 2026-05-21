"""Hybrid retrieval combining BM25 and dense retrieval."""
import heapq
import json
import logging
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.core.document import Document
from src.core.interfaces import BaseEmbedder, BaseRetriever
from .naive import NaiveRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25 (sparse) and dense retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
    """
    
    def __init__(
        self, 
        embedder: BaseEmbedder, 
        top_k: int = 5,
        collection_name: str = "nutrition_docs",
        persist_directory: str = "./chroma_db",
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        **kwargs: Any
    ):
        super().__init__(embedder=embedder, top_k=top_k, **kwargs)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.persist_directory = persist_directory
        
        # Dense retriever
        self._dense_retriever = NaiveRetriever(
            embedder=embedder,
            top_k=top_k * 2,  # Retrieve more for fusion
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        
        # BM25 components
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

        # Persist BM25 corpus per collection, so hybrid retrieval works across restarts.
        # Use the dense retriever's final collection name, which is already unique.
        safe_collection = getattr(self._dense_retriever, "_collection_name", collection_name)
        self._bm25_store_path = Path(persist_directory) / f"{safe_collection}__bm25.jsonl"
        self._load_bm25_store()
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        # Basic tokenization - lowercase and split on whitespace/punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _build_bm25_index(self) -> None:
        """Build BM25 index from stored documents."""
        if not self._documents:
            return
        
        self._tokenized_corpus = [self._tokenize(doc.content) for doc in self._documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def _load_bm25_store(self) -> None:
        """Load persisted BM25 corpus from disk and rebuild index."""
        if not self._bm25_store_path.exists():
            return

        docs: list[Document] = []
        try:
            with self._bm25_store_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    docs.append(Document(
                        id=row.get("id"),
                        content=row.get("content", ""),
                        metadata=row.get("metadata") or {},
                    ))
        except (json.JSONDecodeError, OSError) as e:
            logging.warning("Failed to load BM25 store from %s: %s", self._bm25_store_path, e)
            return

        self._documents = docs
        self._build_bm25_index()

    def _append_to_bm25_store(self, documents: list[Document]) -> None:
        """Append new documents to the BM25 corpus on disk.

        Writes only the new documents that aren't already persisted.
        """
        if not documents:
            return

        self._bm25_store_path.parent.mkdir(parents=True, exist_ok=True)
        with self._bm25_store_path.open("a", encoding="utf-8") as f:
            for doc in documents:
                payload = {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to both BM25 and dense indices.

        Args:
            documents: Documents to index.
        """
        # Add to dense retriever
        self._dense_retriever.add_documents(documents)

        # Add to BM25 index (deduplicate by id)
        existing_ids = {doc.id for doc in self._documents}
        new_docs = [d for d in documents if d.id not in existing_ids]
        self._documents.extend(new_docs)
        self._append_to_bm25_store(new_docs)
        self._build_bm25_index()
    
    def _bm25_retrieve(self, query: str, top_k: int) -> list[Document]:
        """Retrieve using BM25."""
        if self._bm25 is None or not self._documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top-k indices using heap (O(n log k) instead of O(n log n))
        top_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])
        
        results = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append(Document(
                id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=scores[idx],
            ))
        
        return results
    
    def _reciprocal_rank_fusion(
        self, 
        bm25_results: list[Document], 
        dense_results: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank)) for each result list
        """
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        
        # Score BM25 results
        for rank, doc in enumerate(bm25_results):
            rrf_score = self.bm25_weight * (1 / (k + rank + 1))
            doc_scores[doc.id] = doc_scores.get(doc.id, 0) + rrf_score
            doc_map[doc.id] = doc
        
        # Score dense results
        for rank, doc in enumerate(dense_results):
            rrf_score = self.dense_weight * (1 / (k + rank + 1))
            doc_scores[doc.id] = doc_scores.get(doc.id, 0) + rrf_score
            doc_map[doc.id] = doc
        
        # Sort by combined score
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc.score = doc_scores[doc_id]
            results.append(doc)
        
        return results
    
    def retrieve_with_query_embedding(
        self, query: str, query_embedding: list[float], top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve using hybrid BM25 + dense retrieval with a pre-computed embedding."""
        if top_k is None:
            top_k = self.top_k

        fetch_k = top_k * 2

        bm25_results = self._bm25_retrieve(query, fetch_k)
        dense_results = self._dense_retriever.query_by_embedding(query_embedding, fetch_k)

        fused_results = self._reciprocal_rank_fusion(bm25_results, dense_results)
        return fused_results[:top_k]

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve using hybrid BM25 + dense retrieval.

        Args:
            query: Search query.
            top_k: Number of documents to retrieve.

        Returns:
            List of relevant documents with fused scores.
        """
        if top_k is None:
            top_k = self.top_k

        # Get results from both retrievers
        fetch_k = top_k * 2  # Fetch more for better fusion

        bm25_results = self._bm25_retrieve(query, fetch_k)
        dense_results = self._dense_retriever.retrieve(query, fetch_k)

        # Fuse results
        fused_results = self._reciprocal_rank_fusion(bm25_results, dense_results)

        return fused_results[:top_k]
    
    def clear(self) -> None:
        """Clear all documents."""
        self._dense_retriever.clear()
        self._documents = []
        self._bm25 = None
        self._tokenized_corpus = []

        if self._bm25_store_path.exists():
            try:
                self._bm25_store_path.unlink()
            except Exception:
                pass
