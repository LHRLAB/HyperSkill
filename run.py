"""
HyperSkill unified launcher.

One-line commands run HyperSkill on all 11 datasets in parallel
(one subprocess per dataset):

    python run.py --train          # evolve the skill library of every dataset
    python run.py --eval           # evaluate the skill-augmented policy on every dataset

Single-dataset usage:

    python run.py --train --dataset AIME2024
    python run.py --eval  --dataset OmniVideoBench
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from hyperskill.datasets.loaders import DATASET_KEYS

PROJECT_ROOT = Path(__file__).resolve().parent
ALL_DATASETS = list(DATASET_KEYS.keys())


def _spawn(ds_name: str, mode_flag: str) -> subprocess.Popen:
    """Spawn one Python subprocess to train/eval a single dataset."""
    cmd = [sys.executable, "-m", "hyperskill.runner", mode_flag, "--dataset", ds_name]
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{mode_flag.lstrip('-')}_{ds_name}.log"
    f = open(log_file, "w", buffering=1)
    print(f"[launch] {ds_name:<18}  ->  {log_file}", flush=True)
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT)


def _run_parallel(mode_flag: str, datasets: list[str]) -> int:
    procs = [(ds, _spawn(ds, mode_flag)) for ds in datasets]
    failed = 0
    for ds, p in procs:
        rc = p.wait()
        status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"[done]   {ds:<18}  {status}", flush=True)
        if rc != 0:
            failed += 1
    return failed


def main() -> None:
    ap = argparse.ArgumentParser(description="HyperSkill launcher")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true",
                      help="Evolve the skill library (writes expr/skill_library/<dataset>/)")
    mode.add_argument("--eval",  action="store_true",
                      help="Evaluate on the test split (writes outputs/hyperskill_<dataset>/eval_results.json)")
    ap.add_argument("--dataset", default=None, choices=ALL_DATASETS, metavar="DATASET",
                    help="Single dataset (default: all 11 in parallel): " + ", ".join(ALL_DATASETS))
    args = ap.parse_args()

    mode_flag = "--train" if args.train else "--eval"
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    t0 = time.time()
    failed = _run_parallel(mode_flag, datasets)
    print(f"\n[summary] {len(datasets)} dataset(s), {failed} failed, "
          f"{time.time() - t0:.1f}s elapsed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
