"""
Evaluation script for HyperSkill.

Usage:
    # Evaluate one dataset
    python scripts/evaluate.py --dataset dataset/Text/AIME2024

    # Zero-shot baseline (no skill retrieval)
    python scripts/evaluate.py --dataset dataset/VL-Web/FVQA --no-skills

    # Evaluate all datasets
    python scripts/evaluate.py --all

    # Quick eval on a subset
    python scripts/evaluate.py --dataset dataset/Omni/OmniVideoBench --max-samples 50
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperskill.config import HyperSkillConfig
from hyperskill.datasets.loaders import HyperSkillDataset, HyperSkillDataLoader, DATASET_KEYS
from hyperskill.grpo.trainer import HyperSkillGRPOTrainer
from hyperskill.grpo.rewards import compute_f1_reward, compute_format_reward


ALL_DATASETS = [f"dataset/{v['folder']}" for v in DATASET_KEYS.values()]


def _cfg_from_dataset(dataset_path: str) -> tuple[HyperSkillConfig, str]:
    ds_name = Path(dataset_path).name
    info = DATASET_KEYS.get(ds_name)
    if info is None:
        raise ValueError(f"Unknown dataset '{ds_name}'. Available: {sorted(DATASET_KEYS)}")

    cfg = HyperSkillConfig()
    cfg.data.modality = info["modality"]
    cfg.mode = "test"
    cfg.hypergraph.working_dir = f"expr/skill_library/{ds_name}"
    cfg.exp_name = f"hyperskill_{ds_name.lower()}"

    key_file = Path("api_key_apiyi.txt")
    if key_file.exists():
        cfg.llm.api_key = key_file.read_text(encoding="utf-8").strip()

    return cfg, ds_name


async def evaluate_one(
    dataset_path: str,
    split: str,
    no_skills: bool,
    max_samples: int | None,
):
    cfg, ds_name = _cfg_from_dataset(dataset_path)

    print(f"\n{'='*60}")
    print(f"[Eval] Dataset:  {dataset_path}")
    print(f"[Eval] Modality: {cfg.data.modality}")
    print(f"[Eval] Skills:   {'disabled' if no_skills else 'enabled'}")
    print(f"{'='*60}")

    dataset = HyperSkillDataset(
        ds_name,
        split=split,
        max_samples=max_samples,
        dataset_root=Path(dataset_path).parent.parent,
    )
    loader = HyperSkillDataLoader(dataset, batch_size=4, shuffle=False)
    print(f"[Eval] {split} set: {len(dataset)} samples")

    trainer = HyperSkillGRPOTrainer(cfg)

    if no_skills:
        metrics = await _evaluate_zero_shot(trainer, loader, cfg)
    else:
        metrics = await trainer.evaluate(loader)

    metrics["dataset"] = dataset_path
    metrics["modality"] = cfg.data.modality
    metrics["no_skills"] = no_skills

    print(f"\n[Eval] Results for {ds_name}:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    return metrics


async def _evaluate_zero_shot(trainer, loader, cfg):
    """Evaluate without skill augmentation."""
    import base64
    import numpy as np
    from hyperskill.prompts import build_initial_prompt
    from hyperskill.llm import async_llm_call, async_llm_call_with_images

    all_f1 = []
    all_format = []

    for batch in loader:
        for item in batch:
            prompt = build_initial_prompt(
                question=item.question,
                modality=cfg.data.modality,
                task_type=item.task_type,
            )
            image_b64 = [base64.b64encode(img).decode() for img in (item.images or [])]
            if image_b64:
                response = await async_llm_call_with_images(
                    prompt=prompt,
                    image_b64_list=image_b64,
                    cfg=cfg.llm,
                    max_tokens=cfg.rollout.max_new_tokens,
                )
            else:
                response = await async_llm_call(
                    prompt=prompt,
                    cfg=cfg.llm,
                    use_cache=False,
                    max_tokens=cfg.rollout.max_new_tokens,
                )

            f1 = compute_f1_reward(response, item.gold_answers)
            fmt = compute_format_reward([response], item.task_type)
            all_f1.append(f1)
            all_format.append(fmt)

    return {
        "f1": float(np.mean(all_f1)) if all_f1 else 0.0,
        "format": float(np.mean(all_format)) if all_format else 0.0,
        "n_samples": len(all_f1),
    }


async def evaluate_all(split: str, no_skills: bool, max_samples: int | None):
    all_results = []
    for ds_path in ALL_DATASETS:
        if not Path(ds_path).exists():
            print(f"[Eval] Not found: {ds_path}, skipping.")
            continue
        try:
            metrics = await evaluate_one(ds_path, split, no_skills, max_samples)
            if metrics:
                all_results.append(metrics)
        except Exception as e:
            print(f"[Eval] ERROR on {ds_path}: {e}")

    print("\n" + "="*65)
    print(f"{'Dataset':<35} {'Modality':<8} {'F1':>8} {'Format':>8}")
    print("-"*65)
    for r in all_results:
        name = Path(r.get("dataset", "")).name
        modality = r.get("modality", "")
        f1 = r.get("f1", 0.0)
        fmt = r.get("format", 0.0)
        print(f"{name:<35} {modality:<8} {f1:>8.4f} {fmt:>8.4f}")
    print("="*65)

    out_path = "outputs/eval_results.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Eval] Results saved to: {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HyperSkill")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset folder path, e.g. dataset/Omni/OmniVideoBench")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all datasets")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--no-skills", action="store_true",
                        help="Disable skill retrieval (zero-shot baseline)")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if args.all:
        asyncio.run(evaluate_all(args.split, args.no_skills, args.max_samples))
    elif args.dataset:
        asyncio.run(evaluate_one(args.dataset, args.split, args.no_skills, args.max_samples))
    else:
        parser.error("Specify --dataset <path> or --all")
