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
_st_models: dict[str, Any] = {}
_st_lock = threading.Lock()


def _get_st_model(model_name: str = "intfloat/multilingual-e5-base"):
    if model_name in _st_models:
        return _st_models[model_name]
    with _st_lock:
        if model_name in _st_models:
            return _st_models[model_name]
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _st_models[model_name] = model
        return model


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
    import gc
    import torch

    P, R, F1 = bert_score.score(
        [response],
        [reference],
        model_type=model_type,
        lang="pt",
        verbose=False,
    )
    p_val, r_val, f1 = P.item(), R.item(), F1.item()
    del P, R, F1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return f1, {
        "precision": p_val,
        "recall": r_val,
        "model": model_type,
    }


def compute_bertscore_batch(
    responses: list[str],
    references: list[str],
    model_type: str = "xlm-roberta-base",
) -> list[tuple[float, dict[str, Any]]]:
    """BERTScore F1 for multiple pairs in one forward pass (one model load)."""
    import bert_score
    import gc
    import torch

    valid_mask = [bool(r) and bool(ref) for r, ref in zip(responses, references)]
    valid_responses = [r for r, m in zip(responses, valid_mask) if m]
    valid_references = [ref for ref, m in zip(references, valid_mask) if m]

    results: list[tuple[float, dict[str, Any]]] = []
    if valid_responses:
        P, R, F1 = bert_score.score(
            valid_responses,
            valid_references,
            model_type=model_type,
            lang="pt",
            verbose=False,
        )
        for i in range(len(valid_responses)):
            results.append((F1[i].item(), {
                "precision": P[i].item(),
                "recall": R[i].item(),
                "model": model_type,
            }))
        del P, R, F1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Re-insert zero scores for empty pairs
    out: list[tuple[float, dict[str, Any]]] = []
    vi = 0
    for m in valid_mask:
        if m:
            out.append(results[vi])
            vi += 1
        else:
            out.append((0.0, {"reason": "empty input"}))
    return out


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
    # BERTScore first (heavy GPU), then free memory before sentence-transformer
    bertscore, _ = compute_bertscore(response, reference, model_type=bertscore_model)
    sem_sim, _ = compute_semantic_similarity(response, reference, model_name=st_model_name)

    return DeterministicMetrics(
        bleu=bleu,
        rouge_l=rouge_l,
        meteor=meteor,
        bertscore_f1=bertscore,
        semantic_similarity=sem_sim,
    )


def compute_deterministic_metrics_batch(
    responses: list[str],
    references: list[str],
    bertscore_model: str = "xlm-roberta-base",
    st_model_name: str = "intfloat/multilingual-e5-base",
) -> list[DeterministicMetrics]:
    """Batch version: runs BERTScore once for all pairs, then lightweight metrics.

    This avoids loading/unloading the BERTScore model per sample, keeping GPU
    memory usage to a single peak instead of N repeated allocations.
    """
    # 1) BERTScore — batch (one model load, then free GPU)
    bs_results = compute_bertscore_batch(responses, references, model_type=bertscore_model)

    # 2) Lightweight per-sample metrics + sentence-transformer (cached on GPU)
    results: list[DeterministicMetrics] = []
    for resp, ref, (bs_f1, _) in zip(responses, references, bs_results):
        if not resp or not ref:
            results.append(DeterministicMetrics(0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        ref_tokens = _tokenize(ref)
        resp_tokens = _tokenize(resp)
        bleu, _ = compute_bleu(resp, ref, ref_tokens=ref_tokens, resp_tokens=resp_tokens)
        rouge_l, _ = compute_rouge_l(resp, ref)
        meteor, _ = compute_meteor(resp, ref, ref_tokens=ref_tokens, resp_tokens=resp_tokens)
        sem_sim, _ = compute_semantic_similarity(resp, ref, model_name=st_model_name)
        results.append(DeterministicMetrics(
            bleu=bleu,
            rouge_l=rouge_l,
            meteor=meteor,
            bertscore_f1=bs_f1,
            semantic_similarity=sem_sim,
        ))
    return results
