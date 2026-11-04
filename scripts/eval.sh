#!/bin/bash
# Evaluate the trained libraries on the test questions (1 rollout per question, temperature 0, seed 1;
# settings in configs/final_configs.json), then score with both scorers (scripts/score.py).
#
#   bash scripts/eval.sh                        # all 11 datasets in parallel
#   bash scripts/eval.sh AIME2024 WorldSense    # selected datasets
#
# Reads expr/${TAG}_skill_library/<dataset>/, writes outputs/${TAG}_hyperskill_<dataset>/eval_results.json.
# Needs OPENROUTER_API_KEY (paid API calls).
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:-hs}
export HYPERSKILL_DISABLE_LLM_CACHE=1 HYPERSKILL_DISABLE_SEARCH_CACHE=1
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set" >&2; exit 1; }
if [ $# -eq 0 ]; then
  python run.py --eval --tag "$TAG" --configs configs/final_configs.json || true
else
  for ds in "$@"; do
    python run.py --eval --tag "$TAG" --configs configs/final_configs.json --dataset "$ds" &
  done
  wait
fi
python scripts/score.py --tag "$TAG"
