"""
Dataset loaders for HyperSkill.

Dataset folder layout (see DATASET_KEYS for each <group>/<name>):

  dataset/<group>/<name>/
    train.jsonl      one JSON object per line: question, gold_answers,
    test.jsonl       data_source, images / video / audio (relative paths)
    resources/       images / PDFs / videos / audio referenced above

train.jsonl (128 questions per dataset) and test.jsonl are used as is.

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
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# ds_key → folder name under dataset/
# resource_type: "none" | "image" | "pdf" | "video" | "video_audio"
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
    "WorldSense":      {"folder": "Omni/WorldSense",        "modality": "omni", "task_type": "mmreasoning", "resource_type": "video_audio"},
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
    task_type:     str               # e.g. "websearch", "math", "mmreasoning", "docsearch"
    resource_type: str = "none"      # "none" | "image" | "pdf" | "video" | "video_audio"
    # resolved file bytes loaded from resource_paths
    images:        List[bytes]       = field(default_factory=list)
    audio:         Optional[bytes]   = None
    video_path:    Optional[str]     = None
    resource_paths: List[str]        = field(default_factory=list)
    dataset_dir:   str               = ""  # path to dataset folder (for corpus/media search)


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

    def __init__(self, dataset_name: str, split: str = "train"):
        import json as _json
        info = DATASET_KEYS.get(dataset_name)
        if info is None:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                             f"Available: {sorted(DATASET_KEYS)}")

        self._folder = DATASET_ROOT / info["folder"]
        jsonl_path   = self._folder / f"{split}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"{jsonl_path} not found. See README (Data) for the expected layout."
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

        source = str(row.get("data_source", "unknown"))
        # task_type / resource_type: DATASET_KEYS is canonical
        # (task_type stored in a JSONL row is ignored)
        task_type     = self._info["task_type"]
        resource_type = self._info.get("resource_type", "none")

        images:     List[bytes] = []
        audio:      Optional[bytes] = None
        video_path: Optional[str]   = None
        resource_paths: List[str]   = []

        if resource_type == "image":
            for rel in (row.get("images") or []):
                abs_p = self._folder / rel
                resource_paths.append(rel)
                if abs_p.exists() and abs_p.suffix.lower() in self.IMAGE_EXTS:
                    images.append(abs_p.read_bytes())

        elif resource_type == "pdf":
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

        elif resource_type == "video_audio":
            # Omni-modal (WorldSense): load BOTH video keyframes AND audio.
            # Audio often carries the dialogue/effects the question is about.
            vid_rel = row.get("video")
            if vid_rel:
                abs_p = self._folder / vid_rel
                resource_paths.append(vid_rel)
                if abs_p.exists() and abs_p.suffix.lower() in self.VIDEO_EXTS:
                    video_path = str(abs_p)
            aud_rel = row.get("audio")
            if aud_rel:
                abs_p = self._folder / aud_rel
                resource_paths.append(aud_rel)
                if abs_p.exists() and abs_p.suffix.lower() in self.AUDIO_EXTS:
                    try:
                        raw = abs_p.read_bytes()
                        audio = raw[:10 * 1024 * 1024]
                    except OSError:
                        pass
        # resource_type == "none": text-only

        return DataBatch(
            question=question,
            gold_answers=gold_answers,
            source=source,
            task_type=task_type,
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

