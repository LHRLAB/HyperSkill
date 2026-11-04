"""
HyperSkill unified launcher.

One-line commands run HyperSkill on all 11 datasets in parallel
(one subprocess per dataset):

    python run.py --train --eval   # train every library from scratch, then test it
    python run.py --train          # evolve the skill library of every dataset
    python run.py --eval           # evaluate the skill-augmented policy on every dataset

Single-dataset usage:

    python run.py --train --eval --dataset AIME2024
    python run.py --eval  --dataset OmniVideoBench

With both --train and --eval, each dataset is evaluated as soon as its own
library is trained. `--tag NAME` writes the libraries to expr/NAME_skill_library/
and the results to outputs/NAME_hyperskill_<dataset>/ instead of the defaults.

Training uses the 128 training questions of each dataset, evaluation its test
questions. Per-dataset settings and seeds come from configs/final_configs.json
(--configs); see README.md, "Reproducing the HyperSkill results".
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hyperskill.datasets.loaders import DATASET_KEYS

PROJECT_ROOT = Path(__file__).resolve().parent
ALL_DATASETS = list(DATASET_KEYS.keys())


def _run(ds_name: str, mode_flag: str, extra: list[str], tag: str) -> int:
    """Run one Python subprocess that trains/evaluates a single dataset."""
    cmd = [sys.executable, "-m", "hyperskill.runner", mode_flag, "--dataset", ds_name, *extra]
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{tag}_" if tag else ""
    log_file = log_dir / f"{prefix}{mode_flag.lstrip('-')}_{ds_name}.log"
    print(f"[launch] {mode_flag.lstrip('-'):<5} {ds_name:<18}  ->  {log_file}", flush=True)
    with open(log_file, "w", buffering=1) as f:
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    status = "OK" if rc == 0 else f"FAIL({rc})"
    print(f"[done]   {mode_flag.lstrip('-'):<5} {ds_name:<18}  {status}", flush=True)
    return rc


def _pipeline(ds_name: str, mode_flags: list[str], extra: list[str], tag: str) -> int:
    """Run the requested phases of one dataset in order; stop at the first failure."""
    for mode_flag in mode_flags:
        rc = _run(ds_name, mode_flag, extra, tag)
        if rc != 0:
            return rc
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="HyperSkill launcher")
    ap.add_argument("--train", action="store_true",
                    help="Evolve the skill library (writes expr/skill_library/<dataset>/)")
    ap.add_argument("--eval",  action="store_true",
                    help="Evaluate on the test split (writes outputs/hyperskill_<dataset>/eval_results.json)")
    ap.add_argument("--dataset", default=None, choices=ALL_DATASETS, metavar="DATASET",
                    help="Single dataset (default: all 11 in parallel): " + ", ".join(ALL_DATASETS))
    ap.add_argument("--tag", default="",
                    help="Run name: libraries in expr/<tag>_skill_library/, "
                         "results in outputs/<tag>_hyperskill_<dataset>/")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Use only the first N questions of each split (smoke tests)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Training order and LLM sampling seed (default: the --configs seed, else unseeded)")
    ap.add_argument("--configs", default=str(PROJECT_ROOT / "configs" / "final_configs.json"),
                    help="Per-dataset configs JSON (default: configs/final_configs.json)")
    ap.add_argument("--library-tag", default=None,
                    help="With --eval only: evaluate the libraries of this run tag (default: --tag)")
    args = ap.parse_args()
    if not (args.train or args.eval):
        ap.error("give --train, --eval or both")

    mode_flags = (["--train"] if args.train else []) + (["--eval"] if args.eval else [])
    extra: list[str] = []
    if args.tag:
        extra += ["--tag", args.tag]
    if args.max_samples is not None:
        extra += ["--max-samples", str(args.max_samples)]
    if args.seed is not None:
        extra += ["--seed", str(args.seed)]
    if args.configs:
        extra += ["--configs", str(Path(args.configs).resolve())]
    if args.library_tag is not None:
        if args.train:
            ap.error("--library-tag is for --eval only")
        extra += ["--library-tag", args.library_tag]

    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(datasets)) as pool:
        rcs = list(pool.map(lambda ds: _pipeline(ds, mode_flags, extra, args.tag), datasets))
    failed = sum(1 for rc in rcs if rc != 0)
    print(f"\n[summary] {len(datasets)} dataset(s), {failed} failed, "
          f"{time.time() - t0:.1f}s elapsed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
