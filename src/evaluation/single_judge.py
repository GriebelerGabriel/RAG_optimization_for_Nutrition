"""Single-pass LLM-as-judge evaluator for RAG quality metrics.

Evaluates all 5 RAG-equivalent metrics (faithfulness, answer_relevancy,
context_precision, context_recall, answer_correctness) in a single LLM call
per question, using a rubric-based Portuguese prompt.

Returns scores on a 1-5 Likert scale mapped to 0.0-1.0 for downstream
compatibility with the existing visualisation and analysis pipeline.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

logger = logging.getLogger(__name__)

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness"]

LIKERT_MAP = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}

RUBRIC = """\
Você é um juiz especialista em avaliação de sistemas RAG na área de Nutrição.
Avalie a seguinte interação usando a escala Likert de 1 a 5 para CADA critério.

### Critérios e Escala (1 = péssimo, 5 = excelente)

**faithfulness** – A resposta usa APENAS informações presentes no contexto recuperado? Sem alucinações?
**answer_relevancy** – A resposta é relevante e responde diretamente à pergunta do usuário?
**context_precision** – O contexto recuperado é relevante e específico para a pergunta? Há ruído?
**context_recall** – O contexto recuperado contém a informação necessária para responder completamente?
**answer_correctness** – A resposta está factualmente correta em comparação com a resposta de referência?

### Dados

**Pergunta:**
{question}

**Contexto recuperado:**
{contexts}

**Resposta gerada:**
{answer}

**Resposta de referência (ground truth):**
{ground_truth}

### Instruções

1. Avalie cada critério de 1 a 5 seguindo as definições acima.
2. Responda APENAS com um JSON válido, sem texto adicional.

Formato obrigatório:
{{"faithfulness": <1-5>, "answer_relevancy": <1-5>, "context_precision": <1-5>, "context_recall": <1-5>, "answer_correctness": <1-5>}}"""


@dataclass
class SingleJudgeResult:
    question: str
    response: str
    reference: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "response": self.response,
            "reference": self.reference,
            **{m: getattr(self, m) for m in METRIC_NAMES},
        }


def _parse_scores(raw: str) -> dict[str, float]:
    """Extract metric scores from LLM output, mapping 1-5 to 0.0-1.0."""
    match = re.search(r"\{[^}]+\}", raw)
    if not match:
        return {m: float("nan") for m in METRIC_NAMES}

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return {m: float("nan") for m in METRIC_NAMES}

    scores = {}
    for m in METRIC_NAMES:
        val = parsed.get(m)
        if val is not None:
            try:
                val = int(val)
                scores[m] = LIKERT_MAP.get(val, float("nan"))
            except (ValueError, TypeError):
                scores[m] = float("nan")
        else:
            scores[m] = float("nan")
    return scores


def evaluate_single(
    lc_llm,
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> SingleJudgeResult:
    """Evaluate one QA pair through a single LLM call."""
    from langchain_core.messages import HumanMessage

    ctx_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) if contexts else "(sem contexto)"
    prompt = RUBRIC.format(
        question=question,
        contexts=ctx_text,
        answer=answer,
        ground_truth=ground_truth or "(não fornecida)",
    )

    try:
        response = lc_llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if hasattr(response, "content") else str(response)
        scores = _parse_scores(raw)
    except Exception as e:
        logger.warning("Single-judge evaluation failed for question '%s…': %s", question[:50], e)
        scores = {m: float("nan") for m in METRIC_NAMES}

    return SingleJudgeResult(
        question=question,
        response=answer,
        reference=ground_truth,
        **scores,
    )


def evaluate_batch(
    lc_llm,
    samples: list,
    max_workers: int = 8,
) -> list[SingleJudgeResult]:
    """Evaluate all samples in parallel, one LLM call each."""
    results: dict[int, SingleJudgeResult] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                evaluate_single,
                lc_llm,
                s.question,
                s.response,
                s.retrieved_contexts,
                s.reference,
            ): i
            for i, s in enumerate(samples)
        }

        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning("  Q%d failed: %s", idx, e)
            done = len(results)
            if done % 10 == 0 or done == len(samples):
                logger.info("  %d/%d questions evaluated", done, len(samples))

    return [results[i] for i in range(len(samples)) if i in results]
