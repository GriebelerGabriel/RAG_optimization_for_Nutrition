#!/usr/bin/env bash
# Run single-judge RAG evaluation on all configs in parallel (one process per config).
# Usage: ./scripts/ragas_parallel.sh
#         MAX_PARALLEL=8 ./scripts/ragas_parallel.sh
set -e

export PYTHONPATH=.
PYTHON=".venv/bin/python3"

ALL_CONFIGS=(
    fair_gpt4o_mini fair_llama8b llama_best llama_cs1024
    llama_cs256 llama_cs512 llama_hybrid llama_hyde
    llama_ov0 llama_ov100 llama_temp000 llama_temp030 llama_topk3
)

MAX_PARALLEL=${MAX_PARALLEL:-4}

echo "Single-judge evaluation for ${#ALL_CONFIGS[@]} configs (max $MAX_PARALLEL parallel)..."
echo ""

pids=()
names=()

for name in "${ALL_CONFIGS[@]}"; do
    # Enforce concurrency limit
    while [ ${#pids[@]} -ge $MAX_PARALLEL ]; do
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                wait "${pids[$i]}" 2>/dev/null || true
                unset 'pids[$i]'
                break
            fi
        done
        [ ${#pids[@]} -ge $MAX_PARALLEL ] && sleep 2
    done

    $PYTHON scripts/single_judge_eval.py --config "$name" &
    pids+=($!)
    names+=("$name")
    echo "  Started: $name (PID $!)"
done

echo ""
echo "All configs launched. Waiting for ${#pids[@]} processes..."

fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}" 2>/dev/null; then
        echo "FAILED: ${names[$i]}"
        fail=$((fail + 1))
    fi
done

if [ $fail -gt 0 ]; then
    echo "$fail config(s) failed!"
fi

echo ""
echo "Generating summary..."
$PYTHON scripts/single_judge_eval.py --summary
