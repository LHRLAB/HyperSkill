"""
Dataset loaders for HyperSkill.

Dataset folder layout (built by scripts/prepare_data.py):

  dataset/<name>/
    train.parquet    columns: question, gold_answers, data_source,
    test.parquet              task_type, modality, resource_paths
    resources/       images / videos referenced by resource_paths
    corpus.jsonl     (Doc datasets) document chunks for RAG

Datasets
────────
  Text
    AIME2024, AIME2025   math reasoning  (train: DAPO-Math, test: AIME)
    WebWalkerQA          web agent       (train: AFM-RL,   test: WWalker)
  VL · WebSearch
    FVQA, InfoSeek, SimpleVQA, LiveVQA
  VL · Doc
    DocBench, MMLongBench-Doc
  Omni
    OmniVideoBench, WorldSense
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# ds_key → folder name under dataset/
# resource_type: "none" | "image" | "pdf" | "video" | "audio"
DATASET_KEYS: Dict[str, Dict[str, str]] = {
    # Text ───────────────────────────────────────────────────────────────────
    "AIME2024":        {"folder": "Text/AIME2024",          "modality": "text", "task_type": "math",        "resource_type": "none"},
    "AIME2025":        {"folder": "Text/AIME2025",          "modality": "text", "task_type": "math",        "resource_type": "none"},
    "WebWalkerQA":     {"folder": "Text/WebWalkerQA",       "modality": "text", "task_type": "webagent",    "resource_type": "none"},
    # VL · WebSearch ─────────────────────────────────────────────────────────
    "FVQA":            {"folder": "VL-Web/FVQA",            "modality": "vl",   "task_type": "websearch",   "resource_type": "image"},
    "InfoSeek":        {"folder": "VL-Web/InfoSeek",        "modality": "vl",   "task_type": "websearch",   "resource_type": "image"},
    "SimpleVQA":       {"folder": "VL-Web/SimpleVQA",       "modality": "vl",   "task_type": "websearch",   "resource_type": "image"},
    "LiveVQA":         {"folder": "VL-Web/LiveVQA",         "modality": "vl",   "task_type": "websearch",   "resource_type": "image"},
    # VL · Doc ───────────────────────────────────────────────────────────────
    "DocBench":        {"folder": "VL-Doc/DocBench",        "modality": "vl",   "task_type": "docsearch",   "resource_type": "pdf"},
    "MMLongBench-Doc": {"folder": "VL-Doc/MMLongBench-Doc", "modality": "vl",   "task_type": "docsearch",   "resource_type": "pdf"},
    # Omni ───────────────────────────────────────────────────────────────────
    "OmniVideoBench":  {"folder": "Omni/OmniVideoBench",    "modality": "omni", "task_type": "mmreasoning", "resource_type": "video"},
    "WorldSense":      {"folder": "Omni/WorldSense",        "modality": "omni", "task_type": "mmreasoning", "resource_type": "audio"},
}


DATASET_ROOT = Path(__file__).parent.parent.parent / "dataset"


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DataBatch:
    """A single QA example from any modality."""
    question:      str
    gold_answers:  List[str]
    source:        str               # dataset name (e.g. "FVQA")
    task_type:     str               # e.g. "websearch", "math", "mmreasoning"
    modality:      str               # "text", "vl", "omni"
    resource_type: str               # "none" | "image" | "pdf" | "video" | "audio"
    # resolved file bytes loaded from resource_paths
    images:        List[bytes]       = field(default_factory=list)
    audio:         Optional[bytes]   = None
    video_path:    Optional[str]     = None
    resource_paths: List[str]        = field(default_factory=list)
    dataset_dir:   str               = ""  # path to dataset folder (for corpus/media search)
    metadata:      Dict[str, Any]    = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-dataset folder loader
# ---------------------------------------------------------------------------

class HyperSkillDataset:
    """
    Loads a single dataset from its folder under dataset/<name>/.

    Resolves resource_paths to bytes:
      - .jpg / .png / .webp  → images list
      - .mp4 / .avi / .mov   → video_path (path string, not loaded into RAM)
      - .wav / .mp3 / .flac  → audio bytes
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        dataset_root: Optional[Path] = None,
    ):
        import json as _json
        root = Path(dataset_root) if dataset_root else DATASET_ROOT
        info = DATASET_KEYS.get(dataset_name)
        if info is None:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                             f"Available: {sorted(DATASET_KEYS)}")

        self._folder = root / info["folder"]
        jsonl_path   = self._folder / f"{split}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"{jsonl_path} not found. Run scripts/prepare_data.py first."
            )

        try:
            records = []
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(_json.loads(line))
        except Exception as e:
            raise RuntimeError(f"Failed to load {jsonl_path}: {e}")

        if max_samples is not None:
            records = records[:max_samples]

        self._records = records
        self._info    = info

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> DataBatch:
        return self._row_to_batch(self._records[idx])

    def _row_to_batch(self, row: Dict[str, Any]) -> DataBatch:
        question = str(row.get("question", ""))

        gold_raw = row.get("gold_answers", [])
        if isinstance(gold_raw, (list, tuple)):
            gold_answers = [str(g) for g in gold_raw]
        elif gold_raw is not None:
            gold_answers = [str(gold_raw)]
        else:
            gold_answers = []

        source        = str(row.get("data_source", "unknown"))
        task_type     = str(row.get("task_type",   self._info["task_type"]))
        modality      = str(row.get("modality",    self._info["modality"]))
        resource_type = self._info.get("resource_type", "none")

        images:     List[bytes] = []
        audio:      Optional[bytes] = None
        video_path: Optional[str]   = None
        resource_paths: List[str]   = []

        # Only load the media type that this dataset actually uses
        if resource_type == "image":
            for rel in (row.get("images") or []):
                abs_p = self._folder / rel
                resource_paths.append(rel)
                if abs_p.exists() and abs_p.suffix.lower() in self.IMAGE_EXTS:
                    images.append(abs_p.read_bytes())

        elif resource_type == "pdf":
            # PDFs are indexed by DocSearchTool — just record the paths
            for rel in (row.get("images") or []):
                abs_p = self._folder / rel
                if abs_p.suffix.lower() == ".pdf":
                    resource_paths.append(rel)

        elif resource_type == "video":
            rel = row.get("video") or (row.get("images") or [None])[0]
            if rel:
                abs_p = self._folder / rel
                resource_paths.append(rel)
                if abs_p.exists() and abs_p.suffix.lower() in self.VIDEO_EXTS:
                    video_path = str(abs_p)

        elif resource_type == "audio":
            rel = row.get("audio") or (row.get("images") or [None])[0]
            if rel:
                abs_p = self._folder / rel
                resource_paths.append(rel)
                if abs_p.exists() and abs_p.suffix.lower() in self.AUDIO_EXTS:
                    try:
                        raw = abs_p.read_bytes()
                        # cap at 10 MB to avoid OOM
                        audio = raw[:10 * 1024 * 1024]
                    except OSError:
                        pass

        # resource_type == "none": text-only, nothing to load

        return DataBatch(
            question=question,
            gold_answers=gold_answers,
            source=source,
            task_type=task_type,
            modality=modality,
            resource_type=resource_type,
            images=images,
            audio=audio,
            video_path=video_path,
            resource_paths=resource_paths,
            dataset_dir=str(self._folder),
        )


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class HyperSkillDataLoader:
    """Iterable data loader that yields batches of DataBatch objects."""

    def __init__(
        self,
        dataset: HyperSkillDataset,
        batch_size: int = 4,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.seed       = seed
        self._indices   = list(range(len(dataset)))

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[List[DataBatch]]:
        indices = self._indices.copy()
        if self.shuffle:
            random.Random(self.seed).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start: start + self.batch_size]
            yield [self.dataset[i] for i in batch]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset_name: str,
    split: str = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 42,
    dataset_root: Optional[Path] = None,
) -> HyperSkillDataLoader:
    """
    Create a data loader for one named dataset.

    Args:
        dataset_name : key from DATASET_KEYS (e.g. "FVQA", "AIME2024")
        split        : "train" | "test"
        batch_size   : samples per batch
        shuffle      : whether to shuffle indices each epoch
    """
    ds = HyperSkillDataset(
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        dataset_root=dataset_root,
    )
    return HyperSkillDataLoader(ds, batch_size=batch_size, shuffle=shuffle, seed=seed)


def make_multi_dataloader(
    dataset_names: List[str],
    split: str = "train",
    batch_size: int = 4,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 42,
    dataset_root: Optional[Path] = None,
) -> HyperSkillDataLoader:
    """
    Concatenate multiple datasets into a single DataLoader.
    """
    import itertools

    class _ConcatDataset:
        def __init__(self, datasets: List[HyperSkillDataset]):
            self._ds   = datasets
            self._lens = [len(d) for d in datasets]
            self._cum  = [0] + list(itertools.accumulate(self._lens))

        def __len__(self) -> int:
            return self._cum[-1]

        def __getitem__(self, idx: int) -> DataBatch:
            for i, (lo, hi) in enumerate(zip(self._cum, self._cum[1:])):
                if lo <= idx < hi:
                    return self._ds[i][idx - lo]
            raise IndexError(idx)

    datasets = [
        HyperSkillDataset(n, split=split, max_samples=max_samples,
                          dataset_root=dataset_root)
        for n in dataset_names
    ]
    concat = _ConcatDataset(datasets)
    return HyperSkillDataLoader(concat, batch_size=batch_size,
                                shuffle=shuffle, seed=seed)
