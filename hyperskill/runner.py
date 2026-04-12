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
import sys
import time
from pathlib import Path

from .config import HyperSkillConfig, load_config, CONFIGS_DIR
from .datasets.loaders import (
    HyperSkillDataset,
    HyperSkillDataLoader,
    DATASET_KEYS,
    DATASET_ROOT,
)
from .grpo.trainer import HyperSkillGRPOTrainer


_MODALITY_YAML = {
    "text": CONFIGS_DIR / "text.yaml",
    "vl":   CONFIGS_DIR / "vl.yaml",
    "omni": CONFIGS_DIR / "omni.yaml",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_cfg(ds_name: str, mode: str) -> HyperSkillConfig:
    info = DATASET_KEYS.get(ds_name)
    if info is None:
        raise ValueError(f"Unknown dataset '{ds_name}'. Known: {sorted(DATASET_KEYS)}")

    yaml_path = _MODALITY_YAML.get(info["modality"])
    cfg = load_config(str(yaml_path)) if yaml_path and yaml_path.exists() else HyperSkillConfig()

    cfg.data.modality        = info["modality"]
    cfg.data.dataset_names   = [ds_name]
    cfg.hypergraph.working_dir = str(PROJECT_ROOT / "expr" / "skill_library" / ds_name)
    cfg.exp_name             = f"hyperskill_{ds_name.lower()}"
    cfg.mode                 = mode

    key_file = PROJECT_ROOT / "api_key_apiyi.txt"
    if key_file.exists():
        cfg.llm.api_key = key_file.read_text(encoding="utf-8").strip()

    return cfg


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

async def train_one(
    ds_name: str,
    epochs: int = 1,
    resume: bool = False,
    max_samples: int | None = None,
    batch_size: int = 4,
) -> dict:
    cfg = _build_cfg(ds_name, mode="train")

    wd = Path(cfg.hypergraph.working_dir)
    if not resume and wd.exists():
        shutil.rmtree(wd)
        print(f"[{ds_name}] Reset: removed {wd}", flush=True)

    print(f"[{ds_name}] Train · modality={cfg.data.modality} · epochs={epochs}", flush=True)
    dataset = HyperSkillDataset(ds_name, split="train",
                                max_samples=max_samples, dataset_root=DATASET_ROOT)
    loader  = HyperSkillDataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[{ds_name}] {len(dataset)} samples · {len(loader)} batches/epoch", flush=True)

    trainer = HyperSkillGRPOTrainer(cfg)
    last_metrics: dict = {}
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        metrics = await trainer.train(loader)
        elapsed = time.time() - t0
        trainer.skill_library.save()
        print(f"[{ds_name}] Epoch {epoch}/{epochs}: F1={metrics.get('avg_f1', 0):.4f} "
              f"skills={metrics.get('skill_nodes', 0)} elapsed={elapsed:.1f}s", flush=True)
        last_metrics = metrics

        ckpt_dir = PROJECT_ROOT / cfg.output_dir / cfg.exp_name / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return last_metrics


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

async def evaluate_one(
    ds_name: str,
    split: str = "test",
    max_samples: int | None = None,
    batch_size: int = 4,
) -> dict:
    cfg = _build_cfg(ds_name, mode="test")

    print(f"[{ds_name}] Eval · modality={cfg.data.modality} · split={split}", flush=True)
    dataset = HyperSkillDataset(ds_name, split=split,
                                max_samples=max_samples, dataset_root=DATASET_ROOT)
    loader  = HyperSkillDataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"[{ds_name}] {len(dataset)} samples", flush=True)

    trainer = HyperSkillGRPOTrainer(cfg)
    metrics = await trainer.evaluate(loader)
    metrics["dataset"]  = ds_name
    metrics["modality"] = cfg.data.modality

    out_dir = PROJECT_ROOT / cfg.output_dir / cfg.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2))

    print(f"[{ds_name}] F1={metrics.get('f1', 0):.4f} (n={metrics.get('n_samples', 0)})", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# CLI (single-dataset worker — used by run.py to spawn subprocesses)
# ---------------------------------------------------------------------------

def _main():
    p = argparse.ArgumentParser(description="HyperSkill single-dataset worker")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--eval",  action="store_true")
    p.add_argument("--dataset",     required=True, help="Dataset key, e.g. AIME2024")
    p.add_argument("--epochs",      type=int, default=1)
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--split",       default="test")
    args = p.parse_args()

    if args.train:
        asyncio.run(train_one(args.dataset, epochs=args.epochs, resume=args.resume,
                              max_samples=args.max_samples, batch_size=args.batch_size))
    else:
        asyncio.run(evaluate_one(args.dataset, split=args.split,
                                 max_samples=args.max_samples, batch_size=args.batch_size))


if __name__ == "__main__":
    _main()
