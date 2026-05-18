"""Deterministic NLP metrics for evaluating answer quality against references.

Provides BLEU, ROUGE-L, METEOR, BERTScore, and Sentence-BERT semantic
similarity — all deterministic (no LLM calls) and suitable for comparing
against RAGAS LLM-as-judge metrics.
"""
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK data bootstrap (runs once)
# ---------------------------------------------------------------------------
_nltk_ready = False
_nltk_lock = threading.Lock()


def _ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    with _nltk_lock:
        if _nltk_ready:
            return
        import nltk

        for pkg in ("punkt_tab", "wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
            except LookupError:
                nltk.download(pkg, quiet=True)
        _nltk_ready = True


# ---------------------------------------------------------------------------
# Sentence-Transformer cache (avoid reloading per question)
# ---------------------------------------------------------------------------
_st_model = None
_st_lock = threading.Lock()


def _get_st_model(model_name: str = "intfloat/multilingual-e5-base"):
    global _st_model
    if _st_model is not None:
        return _st_model
    with _st_lock:
        if _st_model is not None:
            return _st_model
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer(model_name)
        return _st_model


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Tokenize Portuguese text using NLTK."""
    _ensure_nltk()
    import nltk

    return nltk.word_tokenize(text, language="portuguese")


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def compute_bleu(
    response: str,
    reference: str,
    ref_tokens: list[str] | None = None,
    resp_tokens: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """BLEU score (n-gram precision) with smoothing for short texts."""
    if not response or not reference:
        return 0.0, {"reason": "empty input"}

    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    ref_tok = ref_tokens or _tokenize(reference)
    resp_tok = resp_tokens or _tokenize(response)

    score = sentence_bleu(
        [ref_tok],
        resp_tok,
        smoothing_function=SmoothingFunction().method1,
    )
    return score, {"ref_len": len(ref_tok), "resp_len": len(resp_tok)}


def compute_rouge_l(response: str, reference: str) -> tuple[float, dict[str, Any]]:
    """ROUGE-L F1 (longest common subsequence)."""
    if not response or not reference:
        return 0.0, {"reason": "empty input"}

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    result = scorer.score(reference, response)
    return result["rougeL"].fmeasure, {
        "precision": result["rougeL"].precision,
        "recall": result["rougeL"].recall,
    }


def compute_meteor(
    response: str,
    reference: str,
    ref_tokens: list[str] | None = None,
    resp_tokens: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """METEOR score (alignment + synonymy via WordNet)."""
    if not response or not reference:
        return 0.0, {"reason": "empty input"}

    from nltk.translate.meteor_score import meteor_score

    ref_tok = ref_tokens or _tokenize(reference)
    resp_tok = resp_tokens or _tokenize(response)
    score = meteor_score([ref_tok], resp_tok)
    return score, {"ref_len": len(ref_tok), "resp_len": len(resp_tok)}


def compute_bertscore(
    response: str,
    reference: str,
    model_type: str = "xlm-roberta-base",
) -> tuple[float, dict[str, Any]]:
    """BERTScore F1 using a multilingual transformer."""
    if not response or not reference:
        return 0.0, {"reason": "empty input"}

    import bert_score

    P, R, F1 = bert_score.score(
        [response],
        [reference],
        model_type=model_type,
        lang="pt",
        verbose=False,
    )
    f1 = F1.item()
    return f1, {
        "precision": P.item(),
        "recall": R.item(),
        "model": model_type,
    }


def compute_semantic_similarity(
    response: str,
    reference: str,
    model_name: str = "intfloat/multilingual-e5-base",
) -> tuple[float, dict[str, Any]]:
    """Cosine similarity between sentence embeddings.

    Returns a score in [0, 1] mapped from raw cosine [-1, 1] via (cos+1)/2.
    A raw cosine of 0 (orthogonal) maps to 0.5 — this is intentional so the
    score is comparable to BLEU/ROUGE which naturally sit in [0, 1].
    The raw cosine is included in the detail dict for diagnostics.
    """
    if not response or not reference:
        return 0.0, {"reason": "empty input"}

    import numpy as np

    model = _get_st_model(model_name)
    embeddings = model.encode([response, reference])
    cos_sim = float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )
    score = max(0.0, min(1.0, (cos_sim + 1) / 2))
    return score, {"raw_cosine": cos_sim, "model": model_name}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class DeterministicMetrics:
    """Container for deterministic NLP evaluation metrics."""

    bleu: float
    rouge_l: float
    meteor: float
    bertscore_f1: float
    semantic_similarity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "bleu": self.bleu,
            "rouge_l": self.rouge_l,
            "meteor": self.meteor,
            "bertscore_f1": self.bertscore_f1,
            "semantic_similarity": self.semantic_similarity,
        }


def compute_deterministic_metrics(
    response: str,
    reference: str,
    bertscore_model: str = "xlm-roberta-base",
    st_model_name: str = "intfloat/multilingual-e5-base",
) -> DeterministicMetrics:
    """Compute all deterministic metrics for a (response, reference) pair."""
    if not response or not reference:
        return DeterministicMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    # Tokenize once, reuse for BLEU and METEOR
    ref_tokens = _tokenize(reference)
    resp_tokens = _tokenize(response)

    bleu, _ = compute_bleu(response, reference, ref_tokens=ref_tokens, resp_tokens=resp_tokens)
    rouge_l, _ = compute_rouge_l(response, reference)
    meteor, _ = compute_meteor(response, reference, ref_tokens=ref_tokens, resp_tokens=resp_tokens)
    bertscore, _ = compute_bertscore(response, reference, model_type=bertscore_model)
    sem_sim, _ = compute_semantic_similarity(response, reference, model_name=st_model_name)

    return DeterministicMetrics(
        bleu=bleu,
        rouge_l=rouge_l,
        meteor=meteor,
        bertscore_f1=bertscore,
        semantic_similarity=sem_sim,
    )
