"""
Bootstrap the HyperSkill skill library from a dataset (no GRPO update).

Performs a single pass, extracting skills to cold-start the library
before running evolve_skills.py.

Usage:
    python scripts/build_skill_library.py --all               # all datasets, 50 samples each
    python scripts/build_skill_library.py --dataset dataset/Text/AIME2024
    python scripts/build_skill_library.py --dataset dataset/Text/AIME2024 --max-samples 100
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperskill.config import HyperSkillConfig
from hyperskill.datasets.loaders import HyperSkillDataset, HyperSkillDataLoader, DATASET_KEYS
from hyperskill.skill.library import SkillLibrary
from hyperskill.llm import async_llm_call
from hyperskill.prompts import build_initial_prompt
from hyperskill.grpo.rewards import compute_f1_reward

LOG_FILE = "outputs/train_log.jsonl"


def _log(record: dict) -> None:
    os.makedirs("outputs", exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cfg_from_dataset(dataset_path: str) -> tuple[HyperSkillConfig, str]:
    ds_name = Path(dataset_path).name
    info = DATASET_KEYS.get(ds_name)
    if info is None:
        raise ValueError(f"Unknown dataset '{ds_name}'. Available: {sorted(DATASET_KEYS)}")

    cfg = HyperSkillConfig()
    cfg.data.modality = info["modality"]
    cfg.hypergraph.working_dir = f"expr/skill_library/{ds_name}"
    cfg.exp_name = f"hyperskill_{ds_name.lower()}"

    key_file = Path("api_key_apiyi.txt")
    if key_file.exists():
        cfg.llm.api_key = key_file.read_text(encoding="utf-8").strip()

    return cfg, ds_name


async def build_library(dataset_path: str, split: str, max_samples: int | None, reset: bool = False):
    cfg, ds_name = _cfg_from_dataset(dataset_path)

    if reset:
        wd = Path(cfg.hypergraph.working_dir)
        if wd.exists():
            shutil.rmtree(wd)
            print(f"[BuildSkillLibrary] Reset: removed {wd}")

    print(f"[BuildSkillLibrary] Dataset: {dataset_path}  split={split}")

    dataset = HyperSkillDataset(
        ds_name,
        split=split,
        max_samples=max_samples,
        dataset_root=Path(dataset_path).parent.parent,
    )
    loader = HyperSkillDataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False)
    print(f"[BuildSkillLibrary] Loaded {len(dataset)} samples in {len(loader)} batches")

    library = SkillLibrary(cfg.hypergraph, cfg.llm)
    print(f"[BuildSkillLibrary] Initial library: {library.stats()}")

    total_inserted = 0
    total_batches  = len(loader)
    t_start        = time.time()

    for batch_idx, batch in enumerate(loader):
        t_batch    = time.time()
        experiences = []
        batch_f1s   = []
        for item in batch:
            prompt = build_initial_prompt(
                question=item.question,
                modality=cfg.data.modality,
                task_type=item.task_type,
            )
            try:
                trajectory = await async_llm_call(
                    prompt=prompt,
                    cfg=cfg.llm,
                    use_cache=True,
                    max_tokens=512,
                )
            except Exception as e:
                print(f"  [Warning] LLM call failed: {e}")
                trajectory = item.question

            f1 = compute_f1_reward(trajectory, item.gold_answers)
            batch_f1s.append(f1)
            if f1 <= 0.0:
                continue  # only bootstrap from correct/partial trajectories

            experiences.append({
                "trajectory": trajectory,
                "advantage": f1,
                "source": item.source,
                "modality": cfg.data.modality,
            })

        n = await library.batch_evolve(experiences, max_concurrent=4)
        total_inserted += n
        avg_f1 = sum(batch_f1s) / max(1, len(batch_f1s))
        stats  = library.stats()
        print(
            f"  [Batch {batch_idx+1}/{total_batches}] Inserted {n} skills | "
            f"Total: {total_inserted} | F1={avg_f1:.4f}"
        )
        _log({
            "time":         time.time(),
            "dataset":      ds_name,
            "phase":        "bootstrap",
            "batch":        batch_idx + 1,
            "total_batches": total_batches,
            "epoch":        1,
            "total_epochs": 1,
            "avg_f1":       round(avg_f1, 6),
            "avg_format":   0.0,
            "skill_nodes":  stats.get("skill_nodes", total_inserted),
            "total_nodes":  stats.get("total_nodes", total_inserted),
            "elapsed":      round(time.time() - t_batch, 1),
        })

    library.save()
    print(f"\n[BuildSkillLibrary] Done! Total skills: {total_inserted}")
    print(f"[BuildSkillLibrary] Saved to: {cfg.hypergraph.working_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap HyperSkill skill library")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str,
                       help="Path to dataset folder, e.g. dataset/Text/AIME2024")
    group.add_argument("--all", action="store_true",
                       help="Bootstrap all datasets sequentially")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Samples per dataset (default: 50)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true",
                      help="Add to existing skill library (default behaviour)")
    mode.add_argument("--reset", action="store_true",
                      help="Delete existing skill library and bootstrap from scratch")
    args = parser.parse_args()

    # Clear the monitor log at the start of each bootstrap run
    os.makedirs("outputs", exist_ok=True)
    open(LOG_FILE, "w").close()

    if args.all:
        datasets = [f"dataset/{v['folder']}" for v in DATASET_KEYS.values()]
        async def run_all():
            for ds in datasets:
                print(f"\n{'='*60}\n[Bootstrap] {ds}\n{'='*60}")
                try:
                    await build_library(ds, args.split, args.max_samples, args.reset)
                except Exception as e:
                    print(f"  [ERROR] {ds}: {e}")
        asyncio.run(run_all())
    else:
        asyncio.run(build_library(args.dataset, args.split, args.max_samples, args.reset))
