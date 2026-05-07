"""Shared text utilities for evaluation metrics."""
import re

_WORD_PATTERN = re.compile(r'\b\w{4,}\b')


def text_overlap(text_a: str, text_b: str) -> float:
    """Compute word-overlap ratio of *text_b*'s words found in *text_a*.

    Returns the fraction of words (length >= 4) in *text_b* that also appear
    in *text_a*.
    """
    ref_words = set(_WORD_PATTERN.findall(text_b.lower()))
    if not ref_words:
        return 0.0
    ret_words = set(_WORD_PATTERN.findall(text_a.lower()))
    return len(ref_words & ret_words) / len(ref_words)
