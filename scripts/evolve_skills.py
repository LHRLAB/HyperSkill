"""
Self-evolution loop for the HyperSkill skill library.

Usage:
    python scripts/evolve_skills.py --all                          # train all (reset + 1 epoch)
    python scripts/evolve_skills.py --dataset dataset/Text/AIME2024  # train one dataset
    python scripts/evolve_skills.py --all --epochs 3               # 3 epochs
    python scripts/evolve_skills.py --all --resume                 # continue without reset
    python scripts/evolve_skills.py --all --delete-only            # only delete skill libraries
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

from hyperskill.config import HyperSkillConfig, load_config
from hyperskill.datasets.loaders import HyperSkillDataset, HyperSkillDataLoader, DATASET_KEYS
from hyperskill.grpo.trainer import HyperSkillGRPOTrainer

# Modality → YAML config file under configs/
_MODALITY_CONFIG = {
    "text": "configs/text.yaml",
    "vl":   "configs/vl.yaml",
    "omni": "configs/omni.yaml",
}

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

    modality = info["modality"]

    # Load from modality YAML if it exists, otherwise use defaults
    yaml_path = _MODALITY_CONFIG.get(modality, "")
    if yaml_path and Path(yaml_path).exists():
        cfg = load_config(yaml_path)
    else:
        cfg = HyperSkillConfig()

    # Per-dataset overrides (always applied on top of YAML)
    cfg.data.modality = modality
    cfg.data.dataset_names = [ds_name]
    cfg.hypergraph.working_dir = f"expr/skill_library/{ds_name}"
    cfg.exp_name = f"hyperskill_{ds_name.lower()}"

    key_file = Path("api_key_apiyi.txt")
    if key_file.exists():
        cfg.llm.api_key = key_file.read_text(encoding="utf-8").strip()

    return cfg, ds_name


async def evolve(
    dataset_path: str,
    epochs: int,
    resume: bool,
    max_samples: int | None,
    batch_size: int,
):
    cfg, ds_name = _cfg_from_dataset(dataset_path)

    # Default: reset (delete existing library). --resume skips this.
    if not resume:
        wd = Path(cfg.hypergraph.working_dir)
        if wd.exists():
            shutil.rmtree(wd)
            print(f"[HyperSkill] Reset: removed {wd}")

    print("=" * 60)
    print("[HyperSkill] Training-Free GRPO Self-Evolution")
    print(f"  Dataset:  {dataset_path}")
    print(f"  Modality: {cfg.data.modality}")
    print(f"  Epochs:   {epochs}")
    print(f"  Mode:     {'resume' if resume else 'reset'}")
    print("=" * 60)

    dataset = HyperSkillDataset(
        ds_name, split="train",
        max_samples=max_samples,
        dataset_root=Path(dataset_path).parent.parent,
    )
    loader = HyperSkillDataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[HyperSkill] Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    trainer = HyperSkillGRPOTrainer(cfg)
    if resume and not reset:
        print(f"[HyperSkill] Resuming from: {cfg.hypergraph.working_dir}")
    print(f"[HyperSkill] Skill library: {trainer.skill_library.stats()}")

    all_metrics = []
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"[HyperSkill] Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        t_epoch = time.time()

        def _batch_cb(batch_idx, total_batches, batch_metrics, elapsed):
            lib_stats = trainer.skill_library.stats()
            _log({
                "time":            time.time(),
                "dataset":         ds_name,
                "epoch":           epoch,
                "total_epochs":    epochs,
                "batch":           batch_idx + 1,
                "total_batches":   total_batches,
                "phase":           "batch",
                "avg_f1":          batch_metrics.get("avg_f1", 0.0),
                "avg_format":      batch_metrics.get("avg_format", 0.0),
                "skills_inserted": batch_metrics.get("skills_inserted", 0),
                "skill_nodes":     lib_stats.get("skill_nodes", 0),
                "total_nodes":     lib_stats.get("total_nodes", 0),
                "elapsed":         round(elapsed, 1),
            })
            # Periodically flush skill library to disk so /graph can read it live
            if (batch_idx + 1) % 5 == 0:
                trainer.skill_library.save()

        metrics = await trainer.train(loader, batch_callback=_batch_cb)
        metrics["epoch"] = epoch
        all_metrics.append(metrics)
        elapsed = time.time() - t_epoch

        print(f"\n[Epoch {epoch}] Summary:")
        print(f"  Avg F1:      {metrics['avg_f1']:.4f}")
        print(f"  Skill nodes: {metrics.get('skill_nodes', 0)}")
        print(f"  Elapsed:     {elapsed:.1f}s")

        _log({
            "time":         time.time(),
            "dataset":      ds_name,
            "epoch":        epoch,
            "total_epochs": epochs,
            "avg_f1":       metrics["avg_f1"],
            "avg_format":   metrics.get("avg_format", 0.0),
            "skill_nodes":  metrics.get("skill_nodes", 0),
            "total_nodes":  metrics.get("total_nodes", 0),
            "elapsed":      round(elapsed, 1),
        })

        ckpt_dir = os.path.join(cfg.output_dir, cfg.exp_name, f"epoch_{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.skill_library.save()
        with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[HyperSkill] Checkpoint: {ckpt_dir}")

    os.makedirs(os.path.join(cfg.output_dir, cfg.exp_name), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.exp_name, "evolution_summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n[HyperSkill] Self-evolution complete!")
    print(f"  Final skill library: {trainer.skill_library.stats()}")
    print(f"  Top skills by advantage:")
    for skill in trainer.skill_library.top_skills(10):
        print(f"    [{skill.advantage:.3f}] {skill.name} ({skill.modality})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperSkill self-evolution training loop")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", type=str,
                       help="Path to dataset folder, e.g. dataset/Omni/OmniVideoBench")
    group.add_argument("--all", action="store_true",
                       help="Train all datasets sequentially")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="Continue on top of existing skill library (default: reset)")
    parser.add_argument("--delete-only", action="store_true",
                        help="Only delete the skill library, do not train")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training set size")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not args.delete_only:
        os.makedirs("outputs", exist_ok=True)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"reset": True, "time": time.time()}) + "\n")

    if args.delete_only:
        datasets = ([f"dataset/{v['folder']}" for v in DATASET_KEYS.values()]
                    if args.all else [args.dataset])
        for ds in datasets:
            _, ds_name = _cfg_from_dataset(ds)
            wd = Path(f"expr/skill_library/{ds_name}")
            if wd.exists():
                shutil.rmtree(wd)
                print(f"[Delete] removed {wd}")
            else:
                print(f"[Delete] not found: {wd}")
    elif args.all:
        datasets = [f"dataset/{v['folder']}" for v in DATASET_KEYS.values()]
        async def run_all():
            for ds in datasets:
                print(f"\n{'='*60}\n[Evolve] {ds}\n{'='*60}")
                try:
                    await evolve(ds, args.epochs, args.resume,
                                 args.max_samples, args.batch_size)
                except Exception as e:
                    print(f"  [ERROR] {ds}: {e}")
        asyncio.run(run_all())
    else:
        asyncio.run(evolve(args.dataset, args.epochs, args.resume,
                           args.max_samples, args.batch_size))
