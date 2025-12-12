"""Generation-grounded metrics for evaluating answer quality and faithfulness."""
import re
from dataclasses import dataclass, field
from typing import Any

from src.core.document import Document
from src.core.interfaces import BaseLLM


@dataclass
class GenerationMetrics:
    """Container for generation evaluation metrics."""
    faithfulness: float  # Does the answer only use retrieved context?
    answer_correctness: float  # How correct is the answer vs reference?
    abstain_correctness: float  # Did it correctly abstain when context is missing?
    citation_quality: float  # Does it cite sources properly?
    
    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_correctness": self.answer_correctness,
            "abstain_correctness": self.abstain_correctness,
            "citation_quality": self.citation_quality,
        }


@dataclass 
class GenerationEvalResult:
    """Detailed result of generation evaluation for a single question."""
    question: str
    response: str
    ideal_answer: str | None
    retrieved_chunks: list[Document]
    metrics: GenerationMetrics
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_faithfulness_heuristic(
    response: str, 
    context: list[Document],
) -> tuple[float, dict[str, Any]]:
    """Evaluate if the response is grounded in the provided context using heuristics.
    
    This is a simple heuristic-based approach. For production, consider using
    an LLM-as-judge approach.
    
    Args:
        response: The generated response.
        context: The retrieved documents used as context.
        
    Returns:
        Tuple of (faithfulness score 0-1, details dict).
    """
    if not response or not context:
        return 0.0, {"reason": "empty response or context"}
    
    response_lower = response.lower()
    context_text = " ".join([doc.content.lower() for doc in context])
    
    # Check for abstention phrases (model correctly refuses when unsure)
    abstention_phrases = [
        "não encontrei",
        "não está no contexto",
        "não tenho informação",
        "não posso responder",
        "não há informação",
        "nas fontes disponíveis",
    ]
    is_abstention = any(phrase in response_lower for phrase in abstention_phrases)
    
    if is_abstention:
        return 1.0, {"reason": "abstention detected", "is_abstention": True}
    
    # Extract key claims from response (simple sentence splitting)
    sentences = re.split(r'[.!?]', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return 0.5, {"reason": "no substantial claims found"}
    
    # Check how many sentences have overlap with context
    grounded_count = 0
    for sentence in sentences:
        # Check for significant word overlap (simple heuristic)
        sentence_words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
        context_words = set(re.findall(r'\b\w{4,}\b', context_text))
        
        if len(sentence_words) == 0:
            continue
            
        overlap = len(sentence_words & context_words) / len(sentence_words)
        if overlap > 0.3:  # At least 30% word overlap
            grounded_count += 1
    
    score = grounded_count / len(sentences) if sentences else 0.0
    
    return score, {
        "total_claims": len(sentences),
        "grounded_claims": grounded_count,
        "is_abstention": False,
    }


def evaluate_faithfulness_llm(
    response: str,
    context: list[Document],
    llm: BaseLLM,
) -> tuple[float, dict[str, Any]]:
    """Evaluate faithfulness using LLM-as-judge approach.
    
    Args:
        response: The generated response.
        context: The retrieved documents used as context.
        llm: LLM to use for evaluation.
        
    Returns:
        Tuple of (faithfulness score 0-1, details dict).
    """
    if not response or not context:
        return 0.0, {"reason": "empty response or context"}
    
    context_text = "\n\n".join([f"[{i+1}] {doc.content}" for i, doc in enumerate(context)])
    
    eval_prompt = f"""Avalie se a resposta abaixo é fielmente baseada APENAS no contexto fornecido.

Contexto:
{context_text}

Resposta avaliada:
{response}

Instruções:
1. A resposta deve usar APENAS informações do contexto
2. Qualquer informação não presente no contexto é uma violação de faithfulness
3. Se a resposta corretamente diz "não encontrei" quando a informação não está no contexto, isso é CORRETO

Responda APENAS com um JSON no formato:
{{"score": <0.0 a 1.0>, "reason": "<explicação breve>"}}
"""
    
    try:
        result = llm.generate(eval_prompt)
        # Extract JSON from response
        import json
        json_match = re.search(r'\{[^}]+\}', result)
        if json_match:
            parsed = json.loads(json_match.group())
            return float(parsed.get("score", 0.5)), {"reason": parsed.get("reason", ""), "raw": result}
    except Exception as e:
        pass
    
    return 0.5, {"reason": "failed to parse LLM evaluation", "raw": result if 'result' in dir() else ""}


def evaluate_answer_correctness(
    response: str,
    ideal_answer: str | None,
    llm: BaseLLM | None = None,
) -> tuple[float, dict[str, Any]]:
    """Evaluate answer correctness against a reference answer.
    
    Args:
        response: The generated response.
        ideal_answer: The ideal/reference answer (if available).
        llm: Optional LLM for semantic comparison.
        
    Returns:
        Tuple of (correctness score 0-1, details dict).
    """
    if not ideal_answer:
        return -1.0, {"reason": "no ideal answer provided"}  # -1 indicates N/A
    
    if not response:
        return 0.0, {"reason": "empty response"}
    
    # If LLM provided, use semantic comparison
    if llm:
        eval_prompt = f"""Compare a resposta gerada com a resposta ideal e avalie a correção.

Resposta ideal:
{ideal_answer}

Resposta gerada:
{response}

Avalie de 0.0 a 1.0:
- 1.0 = resposta gerada cobre todos os pontos da resposta ideal corretamente
- 0.5 = resposta parcialmente correta
- 0.0 = resposta incorreta ou sem relação

Responda APENAS com um JSON: {{"score": <0.0 a 1.0>, "reason": "<explicação>"}}
"""
        try:
            result = llm.generate(eval_prompt)
            import json
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                parsed = json.loads(json_match.group())
                return float(parsed.get("score", 0.5)), {"reason": parsed.get("reason", "")}
        except:
            pass
    
    # Fallback: simple word overlap heuristic
    response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
    ideal_words = set(re.findall(r'\b\w{4,}\b', ideal_answer.lower()))
    
    if not ideal_words:
        return 0.5, {"reason": "ideal answer too short for comparison"}
    
    overlap = len(response_words & ideal_words) / len(ideal_words)
    return min(overlap, 1.0), {"word_overlap": overlap}


def evaluate_abstain_correctness(
    response: str,
    context: list[Document],
    has_relevant_context: bool,
) -> tuple[float, dict[str, Any]]:
    """Evaluate if the model correctly abstains when context is insufficient.
    
    Args:
        response: The generated response.
        context: Retrieved documents.
        has_relevant_context: Whether the context actually contains relevant info.
        
    Returns:
        Tuple of (abstain correctness score 0-1, details dict).
    """
    response_lower = response.lower()
    
    abstention_phrases = [
        "não encontrei",
        "não está no contexto", 
        "não tenho informação",
        "não posso responder",
        "não há informação",
        "nas fontes disponíveis",
        "não foi possível encontrar",
    ]
    
    did_abstain = any(phrase in response_lower for phrase in abstention_phrases)
    
    # Correct behavior:
    # - has_relevant_context=True AND did_abstain=False -> correct (answered with context)
    # - has_relevant_context=False AND did_abstain=True -> correct (abstained correctly)
    # - has_relevant_context=True AND did_abstain=True -> incorrect (unnecessarily abstained)
    # - has_relevant_context=False AND did_abstain=False -> incorrect (hallucinated)
    
    if has_relevant_context and not did_abstain:
        return 1.0, {"behavior": "correctly_answered", "did_abstain": did_abstain}
    elif not has_relevant_context and did_abstain:
        return 1.0, {"behavior": "correctly_abstained", "did_abstain": did_abstain}
    elif has_relevant_context and did_abstain:
        return 0.5, {"behavior": "unnecessary_abstention", "did_abstain": did_abstain}
    else:  # not has_relevant_context and not did_abstain
        return 0.0, {"behavior": "potential_hallucination", "did_abstain": did_abstain}


def evaluate_citation_quality(
    response: str,
    context: list[Document],
) -> tuple[float, dict[str, Any]]:
    """Evaluate if the response properly cites sources.
    
    Args:
        response: The generated response.
        context: Retrieved documents (with source info).
        
    Returns:
        Tuple of (citation quality score 0-1, details dict).
    """
    if not response or not context:
        return 0.0, {"reason": "empty response or context"}
    
    # Check for citation patterns
    citation_patterns = [
        r'\[Fonte \d+\]',
        r'\[Fonte:\s*[^\]]+\]',
        r'Fonte:\s*\d+',
        r'de acordo com',
        r'segundo',
        r'conforme',
    ]
    
    citations_found = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        citations_found.extend(matches)
    
    # Check if it's an abstention (doesn't need citations)
    abstention_phrases = ["não encontrei", "não está no contexto", "não tenho informação"]
    is_abstention = any(phrase in response.lower() for phrase in abstention_phrases)
    
    if is_abstention:
        return 1.0, {"reason": "abstention - citations not required", "citations": []}
    
    # Simple heuristic: presence of any citation indication
    if citations_found:
        score = min(len(citations_found) / 2, 1.0)  # Cap at 1.0
        return score, {"citations_found": citations_found, "count": len(citations_found)}
    
    return 0.0, {"reason": "no citations found", "citations": []}


def compute_generation_metrics(
    response: str,
    context: list[Document],
    ideal_answer: str | None = None,
    has_relevant_context: bool = True,
    llm: BaseLLM | None = None,
    use_llm_eval: bool = False,
) -> GenerationMetrics:
    """Compute all generation metrics for a single response.
    
    Args:
        response: Generated response.
        context: Retrieved documents used as context.
        ideal_answer: Reference answer if available.
        has_relevant_context: Whether retrieved context contains relevant info.
        llm: LLM for evaluation (optional).
        use_llm_eval: Whether to use LLM-as-judge for evaluation.
        
    Returns:
        GenerationMetrics object.
    """
    # Faithfulness
    if use_llm_eval and llm:
        faithfulness, _ = evaluate_faithfulness_llm(response, context, llm)
    else:
        faithfulness, _ = evaluate_faithfulness_heuristic(response, context)
    
    # Answer correctness
    correctness, _ = evaluate_answer_correctness(response, ideal_answer, llm if use_llm_eval else None)
    if correctness < 0:  # N/A case
        correctness = -1.0
    
    # Abstain correctness
    abstain, _ = evaluate_abstain_correctness(response, context, has_relevant_context)
    
    # Citation quality
    citation, _ = evaluate_citation_quality(response, context)
    
    return GenerationMetrics(
        faithfulness=faithfulness,
        answer_correctness=correctness,
        abstain_correctness=abstain,
        citation_quality=citation,
    )
