"""Single-judge RAG evaluator.

Evaluates 5 RAG quality dimensions in one LLM call per sample:
  faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness

Usage:
    PYTHONPATH=. python scripts/single_judge_eval.py --config fair_gpt4o_mini
    PYTHONPATH=. python scripts/single_judge_eval.py --all
    PYTHONPATH=. python scripts/single_judge_eval.py --summary
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()
load_dotenv(".env.local")

RESULTS_DIR = Path("results")
CACHE_PATH = RESULTS_DIR / "generation_cache.json"

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

JUDGE_PROMPT = """You are an expert RAG evaluation judge. Evaluate the following RAG system output across 5 dimensions.

## Input
**Question:** {question}

**Retrieved Contexts:**
{contexts}

**Generated Answer:**
{answer}

**Reference (Ground Truth) Answer:**
{reference}

## Scoring Criteria (0.0 to 1.0)

1. **faithfulness**: Is the answer grounded ONLY in the retrieved contexts? No external knowledge or hallucination?
   - 1.0 = every claim is supported by the contexts
   - 0.5 = some claims lack support
   - 0.0 = major hallucination or unsupported claims

2. **answer_relevancy**: Does the answer directly address the question? Is it relevant and complete?
   - 1.0 = perfectly addresses the question
   - 0.5 = partially relevant
   - 0.0 = irrelevant

3. **context_precision**: Are the retrieved contexts relevant to the question? Is there noise/irrelevant chunks?
   - 1.0 = all retrieved contexts are highly relevant
   - 0.5 = mix of relevant and irrelevant
   - 0.0 = all contexts are irrelevant

4. **context_recall**: Does the retrieved context contain the information needed to answer the question (as shown in the reference)?
   - 1.0 = all information from the reference is present in the contexts
   - 0.5 = some required information is missing
   - 0.0 = contexts contain none of the needed information

5. **answer_correctness**: How factually correct is the answer compared to the reference?
   - 1.0 = fully correct and matches reference
   - 0.5 = partially correct
   - 0.0 = incorrect

Respond with ONLY a JSON object:
{{"faithfulness": <score>, "answer_relevancy": <score>, "context_precision": <score>, "context_recall": <score>, "answer_correctness": <score>}}"""


class JudgeScores(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    answer_correctness: float = Field(ge=0.0, le=1.0)


def parse_scores(raw: str) -> dict[str, float]:
    """Parse judge response into score dict, with fallback."""
    json_match = re.search(r"\{[^}]+\}", raw)
    if json_match:
        data = json.loads(json_match.group())
        scores = {}
        for m in METRIC_NAMES:
            v = data.get(m, 0.5)
            scores[m] = max(0.0, min(1.0, float(v)))
        return scores
    # Fallback: return neutral scores
    return {m: 0.5 for m in METRIC_NAMES}


def evaluate_sample(
    llm: ChatOpenAI,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
) -> dict[str, float]:
    """Evaluate a single sample using the judge."""
    ctx_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) if contexts else "[No context retrieved]"
    prompt = JUDGE_PROMPT.format(
        question=question,
        contexts=ctx_text,
        answer=answer,
        reference=reference,
    )
    for attempt in range(3):
        try:
            resp = llm.invoke(prompt)
            return parse_scores(resp.content)
        except Exception as e:
            if attempt == 2:
                print(f"    Warning: judge failed after 3 attempts: {e}")
                return {m: 0.5 for m in METRIC_NAMES}
            time.sleep(2)


def evaluate_config(config_name: str):
    """Evaluate all samples for one config."""
    with open(CACHE_PATH) as f:
        cache = json.load(f)

    entry = next((e for e in cache if e["config_name"] == config_name), None)
    if not entry:
        print(f"Config '{config_name}' not found in cache")
        sys.exit(1)

    samples = entry["samples"]
    print(f"[{config_name}] Evaluating {len(samples)} samples...")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=512, request_timeout=120)

    rows = []
    start = time.time()
    for i, s in enumerate(samples):
        scores = evaluate_sample(
            llm,
            question=s["question"],
            answer=s["response"],
            contexts=s["retrieved_contexts"],
            reference=s["reference"],
        )
        rows.append({
            "user_input": s["question"],
            "response": s["response"],
            "reference": s["reference"],
            **scores,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{config_name}] {i+1}/{len(samples)} done ({time.time()-start:.0f}s)")

    # Save detail CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    detail_path = RESULTS_DIR / f"ragas_detail_{config_name}.csv"
    df.to_csv(detail_path, index=False)

    elapsed = time.time() - start
    print(f"[{config_name}] Done in {elapsed:.0f}s → {detail_path}")


def generate_summary():
    """Merge all detail CSVs into a summary."""
    import pandas as pd

    with open(CACHE_PATH) as f:
        cache = json.load(f)

    rows = []
    for entry in cache:
        name = entry["config_name"]
        csv_path = RESULTS_DIR / f"ragas_detail_{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            means = {c: df[c].mean() for c in METRIC_NAMES}
            rows.append({"config": name, "avg_latency_ms": entry["avg_latency_ms"], **means})

    if not rows:
        print("No detail CSVs found. Run evaluations first.")
        sys.exit(1)

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "ragas_benchmark_results.csv", index=False)
    print("\n" + summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Single-judge RAG evaluator")
    parser.add_argument("--config", help="Evaluate a single config")
    parser.add_argument("--all", action="store_true", help="Evaluate all configs")
    parser.add_argument("--summary", action="store_true", help="Generate summary from existing detail CSVs")
    args = parser.parse_args()

    if args.summary:
        generate_summary()
    elif args.config:
        evaluate_config(args.config)
    elif args.all:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        for entry in cache:
            evaluate_config(entry["config_name"])
        generate_summary()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
