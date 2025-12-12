"""Evaluation metrics for RAG system."""
from dataclasses import dataclass
from typing import Any

from src.core.document import Document


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    mrr: float  # Mean Reciprocal Rank
    
    def to_dict(self) -> dict[str, float]:
        return {
            "precision@1": self.precision_at_1,
            "precision@3": self.precision_at_3,
            "precision@5": self.precision_at_5,
            "recall@5": self.recall_at_5,
            "mrr": self.mrr,
        }


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Calculate precision at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank).
        relevant_ids: Set of relevant document IDs.
        k: Number of top results to consider.
        
    Returns:
        Precision at k.
    """
    if k == 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Calculate recall at k.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank).
        relevant_ids: Set of relevant document IDs.
        k: Number of top results to consider.
        
    Returns:
        Recall at k.
    """
    if not relevant_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Calculate Mean Reciprocal Rank.
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by rank).
        relevant_ids: Set of relevant document IDs.
        
    Returns:
        Reciprocal rank (1/rank of first relevant document, or 0 if none found).
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    retrieved_docs: list[Document],
    relevant_ids: set[str],
) -> RetrievalMetrics:
    """Compute all retrieval metrics.
    
    Args:
        retrieved_docs: List of retrieved documents.
        relevant_ids: Set of relevant document IDs.
        
    Returns:
        RetrievalMetrics object.
    """
    retrieved_ids = [doc.id for doc in retrieved_docs]
    
    return RetrievalMetrics(
        precision_at_1=precision_at_k(retrieved_ids, relevant_ids, 1),
        precision_at_3=precision_at_k(retrieved_ids, relevant_ids, 3),
        precision_at_5=precision_at_k(retrieved_ids, relevant_ids, 5),
        recall_at_5=recall_at_k(retrieved_ids, relevant_ids, 5),
        mrr=mean_reciprocal_rank(retrieved_ids, relevant_ids),
    )
