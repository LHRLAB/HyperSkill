"""
HyperSkill train + evaluate runner.

Provides `train_one` and `evaluate_one` async entry points used by the top-level
`run.py` launcher (which spawns one subprocess per dataset for parallelism).

A single dataset can also be invoked directly via:

    python -m hyperskill.runner --train --dataset AIME2024
    python -m hyperskill.runner --eval  --dataset AIME2024
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from .config import HyperSkillConfig, load_config, CONFIGS_DIR
from .datasets.loaders import (
    HyperSkillDataset,
    HyperSkillDataLoader,
    DATASET_KEYS,
)
from .grpo.trainer import HyperSkillGRPOTrainer


_MODALITY_YAML = {
    "text": CONFIGS_DIR / "text.yaml",
    "vl":   CONFIGS_DIR / "vl.yaml",
    "omni": CONFIGS_DIR / "omni.yaml",
}

EPOCHS = 1
BATCH_SIZE = 4

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _enable_cost_log(cfg: HyperSkillConfig, ds_name: str, phase: str) -> None:
    """Per-(dataset, phase) token usage / cost log written by hyperskill/llm.py
    to outputs/<exp_name>/usage_<phase>.json."""
    f = PROJECT_ROOT / cfg.output_dir / cfg.exp_name / f"usage_{phase}.json"
    os.environ["HYPERSKILL_COST_FILE"] = str(f)
    os.environ["HYPERSKILL_COST_DATASET"] = ds_name
    os.environ["HYPERSKILL_COST_PHASE"] = phase


def _build_cfg(ds_name: str, mode: str) -> HyperSkillConfig:
    info = DATASET_KEYS.get(ds_name)
    if info is None:
        raise ValueError(f"Unknown dataset '{ds_name}'. Known: {sorted(DATASET_KEYS)}")

    cfg = load_config(str(_MODALITY_YAML[info["modality"]]), ds_name, mode)
    cfg.data.modality        = info["modality"]
    cfg.data.dataset_names   = [ds_name]
    cfg.hypergraph.working_dir = str(PROJECT_ROOT / "expr" / "skill_library" / ds_name)
    cfg.exp_name             = f"hyperskill_{ds_name.lower()}"
    return cfg


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

async def train_one(ds_name: str) -> dict:
    cfg = _build_cfg(ds_name, mode="train")
    _enable_cost_log(cfg, ds_name, "train")

    wd = Path(cfg.hypergraph.working_dir)
    if wd.exists():
        shutil.rmtree(wd)
        print(f"[{ds_name}] Reset: removed {wd}", flush=True)

    print(f"[{ds_name}] Train · modality={cfg.data.modality} · "
          f"library run {cfg.library_run} · epochs={EPOCHS}", flush=True)
    dataset = HyperSkillDataset(ds_name, split="train")
    loader  = HyperSkillDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[{ds_name}] {len(dataset)} samples · {len(loader)} batches/epoch", flush=True)

    trainer = HyperSkillGRPOTrainer(cfg)
    last_metrics: dict = {}
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        metrics = await trainer.train(loader)
        elapsed = time.time() - t0
        trainer.skill_library.save()
        print(f"[{ds_name}] Epoch {epoch}/{EPOCHS}: F1={metrics.get('avg_f1', 0):.4f} "
              f"skills={metrics.get('skill_nodes', 0)} elapsed={elapsed:.1f}s", flush=True)
        last_metrics = metrics

        ckpt_dir = PROJECT_ROOT / cfg.output_dir / cfg.exp_name / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return last_metrics


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

async def evaluate_one(ds_name: str) -> dict:
    cfg = _build_cfg(ds_name, mode="test")
    _enable_cost_log(cfg, ds_name, "eval")

    print(f"[{ds_name}] HyperSkill · modality={cfg.data.modality} · split=test", flush=True)
    dataset = HyperSkillDataset(ds_name, split="test")
    loader  = HyperSkillDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[{ds_name}] {len(dataset)} samples", flush=True)

    trainer = HyperSkillGRPOTrainer(cfg)
    metrics = await trainer.evaluate(loader)
    metrics["dataset"]  = ds_name
    metrics["modality"] = cfg.data.modality

    out_dir = PROJECT_ROOT / cfg.output_dir / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2))

    print(f"[{ds_name}] HyperSkill F1={metrics.get('f1', 0):.4f} "
          f"(n={metrics.get('n_samples', 0)}; bootstrap 7×85% "
          f"{metrics.get('bootstrap_mean', 0):.4f}±{metrics.get('bootstrap_std', 0):.4f})", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# CLI (single-dataset worker — used by run.py to spawn subprocesses)
# ---------------------------------------------------------------------------

def _main():
    p = argparse.ArgumentParser(description="HyperSkill single-dataset worker")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--eval",  action="store_true")
    p.add_argument("--dataset", required=True, help="Dataset key, e.g. AIME2024")
    args = p.parse_args()

    if args.train:
        asyncio.run(train_one(args.dataset))
    else:
        asyncio.run(evaluate_one(args.dataset))


if __name__ == "__main__":
    _main()
