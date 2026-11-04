"""
HyperSkill train + evaluate runner.

Provides `train_one` and `evaluate_one` async entry points used by the top-level
`run.py` launcher (which spawns one subprocess per dataset for parallelism).

A single dataset can also be invoked directly via:

    python -m hyperskill.runner --train --dataset AIME2024
    python -m hyperskill.runner --eval  --dataset AIME2024

`--tag NAME` keeps a run apart from the default one: the library goes to
expr/NAME_skill_library/<dataset>/ and the results to outputs/NAME_hyperskill_<dataset>/.
`--max-samples N` uses only the first N questions of the split (smoke tests).

Training uses the 128 training questions (train.jsonl), evaluation the test
questions (test.jsonl). The library records the split it was trained on
(library_meta.json).
`--seed S` fixes the training order and the sampling seed of every LLM call
(default: the seed of the --configs entry).
`--configs FILE` (default configs/final_configs.json) gives per-dataset
library_run / train / eval settings and the seed.
`--library-tag NAME` evaluates the library of run NAME (default: --tag).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
DEFAULT_CONFIGS = PROJECT_ROOT / "configs" / "final_configs.json"

TRAIN_SPLIT, EVAL_SPLIT = "train128", "test"
DEFAULT_LOADER_SEED = 42             # training order when no seed is given


def _load_override(configs: Optional[str], ds_name: str) -> Dict[str, Any]:
    if not configs:
        return {}
    data = json.loads(Path(configs).read_text())
    data = data.get("datasets", data)
    return data.get(ds_name, {})


def _install_keyframe_cache() -> None:
    """Memoise video keyframe extraction on disk (cache/keyframes; a pure
    function of the video file, so the frames are identical)."""
    from .grpo.trainer import install_keyframe_cache
    install_keyframe_cache()


def _apply_seed(seed: Optional[int]) -> None:
    """Seed every LLM call of this process (hyperskill/llm.py)."""
    if seed is None:
        os.environ.pop("HYPERSKILL_SEED", None)
    else:
        os.environ["HYPERSKILL_SEED"] = str(int(seed))


def _enable_cost_log(cfg: HyperSkillConfig, ds_name: str, phase: str) -> None:
    """Per-(dataset, phase) token usage / cost log written by hyperskill/llm.py
    to outputs/<exp_name>/usage_<phase>.json."""
    f = PROJECT_ROOT / cfg.output_dir / cfg.exp_name / f"usage_{phase}.json"
    os.environ["HYPERSKILL_COST_FILE"] = str(f)
    os.environ["HYPERSKILL_COST_DATASET"] = ds_name
    os.environ["HYPERSKILL_COST_PHASE"] = phase


def _build_cfg(ds_name: str, mode: str, tag: str = "", library_tag: Optional[str] = None,
               override: Optional[Dict[str, Any]] = None, train_split: str = "train128") -> HyperSkillConfig:
    info = DATASET_KEYS.get(ds_name)
    if info is None:
        raise ValueError(f"Unknown dataset '{ds_name}'. Known: {sorted(DATASET_KEYS)}")

    cfg = load_config(str(_MODALITY_YAML[info["modality"]]), ds_name, mode, override)
    cfg.data.modality        = info["modality"]
    cfg.data.dataset_names   = [ds_name]
    cfg.data.train_split     = train_split
    prefix = f"{tag}_" if tag else ""
    lib_tag = tag if library_tag is None else library_tag
    lib_prefix = f"{lib_tag}_" if lib_tag else ""
    cfg.hypergraph.working_dir = str(PROJECT_ROOT / "expr" / f"{lib_prefix}skill_library" / ds_name)
    cfg.exp_name             = f"{prefix}hyperskill_{ds_name.lower()}"
    return cfg


def _config_digest(cfg: HyperSkillConfig) -> Dict[str, Any]:
    from dataclasses import asdict
    d = asdict(cfg)
    d["hypergraph"].pop("working_dir", None)
    d.pop("exp_name", None)
    return d


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

async def train_one(ds_name: str, tag: str = "", max_samples: int | None = None,
                    seed: Optional[int] = None, configs: Optional[str] = str(DEFAULT_CONFIGS)) -> dict:
    override = _load_override(configs, ds_name)
    if seed is None:
        seed = override.get("seed")
    train_split = TRAIN_SPLIT
    cfg = _build_cfg(ds_name, mode="train", tag=tag, override=override, train_split=train_split)
    _enable_cost_log(cfg, ds_name, "train")
    _apply_seed(seed)
    _install_keyframe_cache()

    wd = Path(cfg.hypergraph.working_dir)
    if wd.exists():
        shutil.rmtree(wd)
        print(f"[{ds_name}] Reset: removed {wd}", flush=True)

    print(f"[{ds_name}] Train · modality={cfg.data.modality} · split={train_split} · "
          f"library run {cfg.library_run} · seed={seed} · epochs={EPOCHS}", flush=True)
    dataset = HyperSkillDataset(ds_name, split=train_split, max_samples=max_samples)
    loader  = HyperSkillDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   seed=DEFAULT_LOADER_SEED if seed is None else int(seed))
    print(f"[{ds_name}] {len(dataset)} samples · {len(loader)} batches/epoch", flush=True)

    trainer = HyperSkillGRPOTrainer(cfg)
    last_metrics: dict = {}
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        metrics = await trainer.train(loader)
        elapsed = time.time() - t0
        trainer.skill_library.save()
        metrics.update(split=train_split, seed=seed, library_run=cfg.library_run,
                       n_train=len(dataset))
        print(f"[{ds_name}] Epoch {epoch}/{EPOCHS}: F1={metrics.get('avg_f1', 0):.4f} "
              f"skills={metrics.get('skill_nodes', 0)} elapsed={elapsed:.1f}s", flush=True)
        last_metrics = metrics

        ckpt_dir = PROJECT_ROOT / cfg.output_dir / cfg.exp_name / f"epoch_{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    meta = {"dataset": ds_name, "train_split": train_split, "seed": seed,
            "library_run": cfg.library_run, "max_samples": max_samples,
            "n_train": len(dataset), "config": _config_digest(cfg)}
    (wd / "library_meta.json").write_text(json.dumps(meta, indent=2))
    return last_metrics


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

async def evaluate_one(ds_name: str, tag: str = "", max_samples: int | None = None,
                       seed: Optional[int] = None, configs: Optional[str] = str(DEFAULT_CONFIGS),
                       library_tag: Optional[str] = None) -> dict:
    override = _load_override(configs, ds_name)
    if seed is None:
        seed = override.get("seed")
    train_split, eval_split = TRAIN_SPLIT, EVAL_SPLIT
    cfg = _build_cfg(ds_name, mode="test", tag=tag, library_tag=library_tag,
                     override=override, train_split=train_split)
    meta_f = Path(cfg.hypergraph.working_dir) / "library_meta.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    if meta and meta.get("train_split") != train_split:
        raise RuntimeError(f"{cfg.hypergraph.working_dir} was trained on {meta.get('train_split')}, "
                           f"not on {train_split}")
    _enable_cost_log(cfg, ds_name, "eval")
    _apply_seed(seed)
    _install_keyframe_cache()

    print(f"[{ds_name}] HyperSkill · modality={cfg.data.modality} · split={eval_split} · "
          f"library {cfg.hypergraph.working_dir} · seed={seed}", flush=True)
    dataset = HyperSkillDataset(ds_name, split=eval_split, max_samples=max_samples)
    loader  = HyperSkillDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[{ds_name}] {len(dataset)} samples", flush=True)

    trainer = HyperSkillGRPOTrainer(cfg)
    metrics = await trainer.evaluate(loader)
    metrics["dataset"]  = ds_name
    metrics["modality"] = cfg.data.modality
    metrics["split"]    = eval_split
    metrics["seed"]     = seed
    metrics["library"]  = cfg.hypergraph.working_dir
    metrics["library_meta"] = meta
    metrics["config"]   = _config_digest(cfg)

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
    p.add_argument("--tag", default="", help="Run name prefix for the library and output dirs")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Use only the first N questions of the split (smoke tests)")
    p.add_argument("--seed", type=int, default=None,
                   help="Training order and LLM sampling seed (default: the --configs seed)")
    p.add_argument("--configs", default=str(DEFAULT_CONFIGS),
                   help="Per-dataset configs JSON (default: configs/final_configs.json)")
    p.add_argument("--library-tag", default=None, help="Evaluate the library of this run tag (default: --tag)")
    args = p.parse_args()

    if args.train:
        asyncio.run(train_one(args.dataset, args.tag, args.max_samples, args.seed, args.configs))
    else:
        asyncio.run(evaluate_one(args.dataset, args.tag, args.max_samples,
                                 args.seed, args.configs, args.library_tag))


if __name__ == "__main__":
    _main()
