#!/bin/bash
# Train the HyperSkill skill libraries (1 epoch over the 128 training questions of each dataset,
# batch size 4, 2 rollouts per question, seed 1; settings in configs/final_configs.json).
#
#   bash scripts/train.sh                       # all 11 datasets in parallel
#   bash scripts/train.sh AIME2024 WorldSense   # selected datasets
#
# Libraries: expr/${TAG}_skill_library/<dataset>/   logs: outputs/logs/   (TAG defaults to "hs")
# Needs OPENROUTER_API_KEY (paid API calls).
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${TAG:-hs}
export HYPERSKILL_DISABLE_LLM_CACHE=1 HYPERSKILL_DISABLE_SEARCH_CACHE=1
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set" >&2; exit 1; }
if [ $# -eq 0 ]; then
  python run.py --train --tag "$TAG" --configs configs/final_configs.json
else
  for ds in "$@"; do
    python run.py --train --tag "$TAG" --configs configs/final_configs.json --dataset "$ds" &
  done
  wait
fi
