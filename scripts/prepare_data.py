#!/usr/bin/env python3
"""
scripts/prepare_data.py — HyperSkill dataset builder

Output layout:
  dataset/
    Text/
      AIME2024/       train.jsonl (1024)  test.jsonl (30)
      AIME2025/       train.jsonl (1024)  test.jsonl (30)
      WebWalkerQA/    train.jsonl (616)   test.jsonl (64)
    VL-Web/
      FVQA/           train.jsonl (1024)  test.jsonl (64)   resources/*.jpg
      InfoSeek/       train.jsonl (1024)  test.jsonl (64)   resources/*.jpg
      SimpleVQA/      train.jsonl (949)   test.jsonl (64)   resources/*.jpg
      LiveVQA/        train.jsonl (1024)  test.jsonl (64)   resources/*.jpg
    VL-Doc/
      DocBench/       train.jsonl (472)   test.jsonl (64)   resources/*.jpg  corpus.jsonl
      MMLongBench-Doc/ train.jsonl (507)  test.jsonl (64)   resources/*.jpg  corpus.jsonl
    Omni/
      OmniVideoBench/ train.jsonl (445)   test.jsonl (64)   resources/*.mp4
      WorldSense/     train.jsonl (1024)  test.jsonl (64)

JSONL schema (every split):
  question      str
  gold_answers  list[str]
  data_source   str
  task_type     str         math | webagent | websearch | mmreasoning
  modality      str         text | vl | omni
  images        list[str]   relative paths from dataset folder  (e.g. "resources/00001.jpg")
  video         str | null  relative path to .mp4 (audio embedded inside)
  audio         str | null  always null — audio is inside the mp4

Usage:
  python scripts/prepare_data.py                     # all datasets
  python scripts/prepare_data.py --category Text     # one category
  python scripts/prepare_data.py --dataset AIME2024  # one dataset
  python scripts/prepare_data.py --verify-only
  python scripts/prepare_data.py --hf-token TOKEN    # for gated HF datasets
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"

# Local MMGraph-R1 sources
HYPEREVO_DIR   = Path("/mnt/hdd2/home/luohaoran/github/MMGraph-R1-datasets/datasets_hyperevo")
MMGRAPH_DIR    = Path("/mnt/hdd2/home/luohaoran/github/MMGraph-R1/datasets")
MMGRAPH_CORPUS = MMGRAPH_DIR / "all_datasets_corpus.jsonl"

SEED = 42

# ── dataset registry ──────────────────────────────────────────────────────────

REGISTRY: Dict[str, Dict] = {
    # ── Text ─────────────────────────────────────────────
    "AIME2024": dict(
        cat="Text", folder="AIME2024",
        train=1024, test=30,
        modality="text", task_type="math",
    ),
    "AIME2025": dict(
        cat="Text", folder="AIME2025",
        train=1024, test=30,
        modality="text", task_type="math",
    ),
    "WebWalkerQA": dict(
        cat="Text", folder="WebWalkerQA",
        train=616, test=64,
        modality="text", task_type="webagent",
    ),
    # ── VL-Web ───────────────────────────────────────────
    "FVQA": dict(
        cat="VL-Web", folder="FVQA",
        train=1024, test=64,
        modality="vl", task_type="websearch",
    ),
    "InfoSeek": dict(
        cat="VL-Web", folder="InfoSeek",
        train=1024, test=64,
        modality="vl", task_type="websearch",
    ),
    "SimpleVQA": dict(
        cat="VL-Web", folder="SimpleVQA",
        train=949, test=64,
        modality="vl", task_type="websearch",
    ),
    "LiveVQA": dict(
        cat="VL-Web", folder="LiveVQA",
        train=1024, test=64,
        modality="vl", task_type="websearch",
    ),
    # ── VL-Doc ───────────────────────────────────────────
    "DocBench": dict(
        cat="VL-Doc", folder="DocBench",
        train=472, test=64,
        modality="vl", task_type="mmreasoning",
    ),
    "MMLongBench-Doc": dict(
        cat="VL-Doc", folder="MMLongBench-Doc",
        train=507, test=64,
        modality="vl", task_type="mmreasoning",
    ),
    # ── Omni ─────────────────────────────────────────────
    "OmniVideoBench": dict(
        cat="Omni", folder="OmniVideoBench",
        train=445, test=64,   # mteb/OmniVideoBench_subset: 509 total
        modality="omni", task_type="mmreasoning",
    ),
    "WorldSense": dict(
        cat="Omni", folder="WorldSense",
        train=1024, test=64,
        modality="omni", task_type="mmreasoning",
    ),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _ds_dir(info: Dict) -> Path:
    return DATASET_ROOT / info["cat"] / info["folder"]


def _make_dirs(info: Dict) -> Tuple[Path, Path]:
    d, res = _ds_dir(info), _ds_dir(info) / "resources"
    d.mkdir(parents=True, exist_ok=True)
    res.mkdir(exist_ok=True)
    return d, res


def _clean(s: Any) -> str:
    s = s if isinstance(s, str) else str(s)
    return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _parse_answers(raw: Any) -> List[str]:
    """Convert any answer representation to list[str]."""
    if isinstance(raw, list):
        return [_clean(a) for a in raw if str(a).strip()]
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                v = ast.literal_eval(raw)
                if isinstance(v, list):
                    return [_clean(a) for a in v if str(a).strip()]
            except Exception:
                pass
        return [_clean(raw)] if raw else []
    return [_clean(str(raw))] if raw is not None else []


def _save_bytes(res_dir: Path, fname: str, data: Any) -> Optional[str]:
    """Write raw bytes to resources/ and return relative path, or None."""
    if isinstance(data, dict):
        data = data.get("bytes")
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    (res_dir / fname).write_bytes(data)
    return f"resources/{fname}"


def _row(question: str, gold_answers: List[str], data_source: str,
         task_type: str, modality: str,
         images: Optional[List[str]] = None,
         video: Optional[str] = None,
         audio: Optional[str] = None) -> Dict:
    return dict(
        question=_clean(question),
        gold_answers=[_clean(a) for a in (gold_answers or [])],
        data_source=data_source,
        task_type=task_type,
        modality=modality,
        images=list(images or []),
        video=video,
        audio=audio,
    )


def _save(out_dir: Path, split: str, rows: List[Dict]) -> None:
    path = out_dir / f"{split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    {path.relative_to(REPO_ROOT)}  ({len(rows)} rows)")


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Text: AIME ────────────────────────────────────────────────────────────────

_DAPO_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
_DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def _dapo_question(item: Dict) -> str:
    """Extract bare math problem from a DAPO-Math-17K item."""
    prompt = item.get("prompt", "")
    if isinstance(prompt, list):
        content = prompt[0].get("content", "") if prompt else ""
    else:
        content = str(prompt)
    # strip wrapper
    if content.startswith(_DAPO_PREFIX):
        content = content[len(_DAPO_PREFIX):]
    if content.endswith(_DAPO_SUFFIX):
        content = content[: -len(_DAPO_SUFFIX)]
    return content.strip()


def _dapo_answer(item: Dict) -> str:
    rm = item.get("reward_model", {})
    if isinstance(rm, dict):
        return str(rm.get("ground_truth", ""))
    return ""


def _load_dapo(n: int, exclude: set, seed: int) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train",
                      trust_remote_code=True)
    items = list(ds)
    random.Random(seed).shuffle(items)
    rows: List[Dict] = []
    for it in items:
        if len(rows) >= n:
            break
        q = _dapo_question(it)
        if not q or q in exclude:
            continue
        rows.append(_row(q, _parse_answers(_dapo_answer(it)),
                         "AIME2024", "math", "text"))
    return rows


def prepare_aime2024(info: Dict) -> None:
    print("\n── AIME 2024")
    from datasets import load_dataset
    out, _ = _make_dirs(info)

    raw      = list(load_dataset("AI-MO/aimo-validation-aime", split="train",
                                 trust_remote_code=True))
    test_rows = [_row(it["problem"], [str(it["answer"])],
                      "AIME2024", "math", "text")
                 for it in raw[: info["test"]]]
    test_qs  = {r["question"] for r in test_rows}

    train_rows = _load_dapo(info["train"], test_qs, SEED)
    # reuse data_source = AIME2024 (already set above)
    _save(out, "train", train_rows)
    _save(out, "test",  test_rows)


def prepare_aime2025(info: Dict) -> None:
    print("\n── AIME 2025")
    from datasets import load_dataset
    out, _ = _make_dirs(info)

    test_rows: List[Dict] = []
    for hf_id, split in [("math-ai/aime25", "test"),
                         ("AI-MO/aimo-validation-aime", "train")]:
        try:
            items = list(load_dataset(hf_id, split=split, trust_remote_code=True))
            # prefer 2025 items if the field exists
            yr = [i for i in items if "2025" in str(i.get("url", i.get("year", "")))]
            items = yr if yr else items
            for it in items[: info["test"]]:
                q = it.get("problem", it.get("question", ""))
                a = str(it.get("answer", ""))
                test_rows.append(_row(q, [a], "AIME2025", "math", "text"))
            if test_rows:
                print(f"    test: {len(test_rows)} items from {hf_id}/{split}")
                break
        except Exception as e:
            print(f"    [skip] {hf_id}: {e}")

    if not test_rows:
        print("    [WARN] AIME 2025 test not found")

    test_qs    = {r["question"] for r in test_rows}
    train_rows = _load_dapo(info["train"], test_qs, SEED + 1)
    for r in train_rows:
        r["data_source"] = "AIME2025"
    _save(out, "train", train_rows)
    _save(out, "test",  test_rows)


# ── Text: WebWalkerQA ─────────────────────────────────────────────────────────

def prepare_webwalkerqa(info: Dict) -> None:
    print("\n── WebWalkerQA")
    from datasets import load_dataset
    out, _ = _make_dirs(info)
    rng     = random.Random(SEED)

    # test split: WebWalkerQA "main"
    wq = list(load_dataset("callanwu/WebWalkerQA", split="main",
                            trust_remote_code=True))
    rng.shuffle(wq)
    test_rows = [_row(it.get("Question", it.get("question", "")),
                      [str(it.get("Answer", it.get("answer", "")))],
                      "WebWalkerQA", "webagent", "text")
                 for it in wq[: info["test"]]]
    test_qs = {r["question"] for r in test_rows}

    # train split: AFM-WebAgent — clean questions are in extra_info
    afm  = list(load_dataset("PersonalAILab/AFM-WebAgent-RL-Dataset", split="train",
                              trust_remote_code=True))
    rng.shuffle(afm)
    train_rows: List[Dict] = []
    for it in afm:
        if len(train_rows) >= info["train"]:
            break
        ei = it.get("extra_info", {})
        if isinstance(ei, str):
            try:
                ei = ast.literal_eval(ei)
            except Exception:
                ei = {}
        q = str(ei.get("question", "")).strip()
        if not q or q in test_qs:
            continue
        a = ei.get("answer", "")
        train_rows.append(_row(q, _parse_answers(a), "WebWalkerQA", "webagent", "text"))
    _save(out, "train", train_rows)
    _save(out, "test",  test_rows)


# ── VL-Web ────────────────────────────────────────────────────────────────────

def _load_hyperevo() -> pd.DataFrame:
    frames = []
    for f in sorted(HYPEREVO_DIR.glob("*.parquet")):
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prepare_vl_web(info: Dict) -> None:
    src = info["folder"]   # same as data_source key in hyperevo
    print(f"\n── {src}")
    out, res = _make_dirs(info)
    rng = random.Random(SEED)

    df = _load_hyperevo()
    sub = df[df["data_source"] == src].reset_index(drop=True)
    if sub.empty:
        print(f"    [ERROR] no rows for {src} in hyperevo parquets")
        return

    idx_all = list(range(len(sub)))
    rng.shuffle(idx_all)

    def _make_row(df_idx: int, tag: str) -> Dict:
        row     = sub.iloc[df_idx]
        q       = _clean(str(row.get("question", "")))
        rm      = row.get("reward_model", {})
        raw_ans = rm.get("candidate_answers", "") if isinstance(rm, dict) else str(rm)
        imgs: List[str] = []
        for i, img in enumerate(row.get("images") or []):
            fname = f"{tag}_{df_idx:05d}_img{i:02d}.jpg"
            p = _save_bytes(res, fname, img)
            if p:
                imgs.append(p)
        return _row(q, _parse_answers(raw_ans), src, info["task_type"], info["modality"],
                    images=imgs)

    n_test  = info["test"]
    n_train = info["train"]
    test_rows  = [_make_row(i, "test")  for i in idx_all[:n_test]]
    train_rows = [_make_row(i, "train") for i in idx_all[n_test: n_test + n_train]]
    _save(out, "train", train_rows)
    _save(out, "test",  test_rows)


# ── VL-Doc ────────────────────────────────────────────────────────────────────

def _mmgraph_rows(src: str) -> List[Dict]:
    """Collect unique QA rows for a given data_source from all MMGraph-R1 parquets."""
    seen: Dict[str, Dict] = {}
    for f in sorted(MMGRAPH_DIR.glob("*.parquet")):
        try:
            df  = pd.read_parquet(f)
            sub = df[df["data_source"] == src]
            for _, row in sub.iterrows():
                q = _clean(str(row.get("question", "")))
                if not q or q in seen:
                    continue
                rm  = row.get("reward_model", {})
                raw = rm.get("candidate_answers", "") if isinstance(rm, dict) else str(rm)
                seen[q] = _row(q, _parse_answers(raw),
                               src, "mmreasoning", "vl")
        except Exception:
            pass
    return list(seen.values())


def _build_corpus(src: str, ds_dir: Path, res_dir: Path) -> None:
    if not MMGRAPH_CORPUS.exists():
        return
    out_path = ds_dir / "corpus.jsonl"
    count    = 0
    with open(MMGRAPH_CORPUS) as fin, open(out_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            if entry.get("corpus_source") != src:
                continue
            img_paths: List[str] = []
            for i, img in enumerate(entry.get("corpus_images") or []):
                fname = f"corpus_{count:05d}_img{i:02d}.jpg"
                p = _save_bytes(res_dir, fname, img)
                if p:
                    img_paths.append(p)
            fout.write(json.dumps({
                "corpus_text":   entry.get("corpus_text", ""),
                "corpus_images": img_paths,
                "corpus_source": src,
            }, ensure_ascii=False) + "\n")
            count += 1
    print(f"    corpus: {count} chunks → {out_path.relative_to(REPO_ROOT)}")


def prepare_vl_doc(info: Dict) -> None:
    src = info["folder"]
    print(f"\n── {src}")
    out, res = _make_dirs(info)
    rng      = random.Random(SEED)
    rows     = _mmgraph_rows(src)
    if not rows:
        print(f"    [ERROR] no rows for {src}"); return
    rng.shuffle(rows)
    n_test  = min(info["test"],  len(rows))
    n_train = min(info["train"], len(rows) - n_test)
    _save(out, "train", rows[:n_train])
    _save(out, "test",  rows[n_train: n_train + n_test])
    _build_corpus(src, out, res)


# ── Omni ──────────────────────────────────────────────────────────────────────

def _omni_row_from(item: Dict, idx: int, res_dir: Path, info: Dict) -> Dict:
    q    = str(item.get("question", item.get("Question", ""))).strip()
    opts = item.get("options", item.get("choices", item.get("candidates", [])))
    if isinstance(opts, list) and opts:
        q += "\n" + "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
    a = item.get("answer", item.get("Answer", ""))

    # save video bytes; audio is already embedded inside the mp4
    video_path = None
    raw_video  = item.get("video")
    if isinstance(raw_video, dict):
        raw_video = raw_video.get("bytes")
    if isinstance(raw_video, (bytes, bytearray)) and raw_video:
        video_path = _save_bytes(res_dir, f"{idx:05d}.mp4", raw_video)

    return _row(q, _parse_answers(a) if not isinstance(a, list) else [str(x) for x in a],
                info["folder"], info["task_type"], info["modality"],
                video=video_path)


def _load_hf_arrow_cache(hf_id: str, split: str = "test") -> Optional[List[Dict]]:
    """Read HF Arrow cache directly (bypasses video decode that requires torch)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    cache_root = Path.home() / ".cache/huggingface/datasets"

    def _to_cache_name(s: str) -> str:
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        return s.lower().replace("/", "___").replace("-", "_")

    matches = list(cache_root.glob(f"{_to_cache_name(hf_id)}*"))
    if not matches:
        return None
    arrow_files = sorted(matches[0].rglob(f"*-{split}-*.arrow"))
    if not arrow_files:
        return None
    rows: List[Dict] = []
    for af in arrow_files:
        try:
            with pa.memory_map(str(af), "r") as src:
                table = ipc.open_stream(src).read_all()
            for i in range(len(table)):
                rows.append({col: table.column(col)[i].as_py()
                              for col in table.schema.names})
        except Exception:
            pass
    return rows or None


def prepare_omni(info: Dict, hf_id: str, split: str = "test",
                 fallback_id: Optional[str] = None,
                 token: Optional[str] = None) -> None:
    from datasets import load_dataset
    print(f"\n── {info['folder']}")
    out, res = _make_dirs(info)

    items: Optional[List] = None

    # 1) try primary HF dataset
    try:
        ds    = load_dataset(hf_id, split=split, token=token, trust_remote_code=True)
        items = list(ds)
    except Exception as e:
        print(f"    [skip] {hf_id}: {e}")

    # 2) try fallback HF dataset
    if items is None and fallback_id:
        for sp in (split, "test", "train"):
            try:
                ds    = load_dataset(fallback_id, split=sp, token=token,
                                     trust_remote_code=True)
                items = list(ds)
                break
            except Exception as e:
                print(f"    [skip] {fallback_id}/{sp}: {e}")

    # 3) try reading from local HF Arrow cache (skips torch video decode)
    if items is None and fallback_id:
        print(f"    Trying Arrow cache for {fallback_id}")
        items = _load_hf_arrow_cache(fallback_id, "test")

    if not items:
        print(f"    [ERROR] no data found for {info['folder']}")
        if token is None:
            print("    Pass --hf-token YOUR_TOKEN for gated datasets")
        return

    random.Random(SEED).shuffle(items)
    n      = info["train"] + info["test"]
    rows   = [_omni_row_from(it, i, res, info) for i, it in enumerate(items[:n])]
    _save(out, "train", rows[:info["train"]])
    _save(out, "test",  rows[info["train"]:])


# ── Verification ──────────────────────────────────────────────────────────────

def verify_all() -> bool:
    print("\n── Verification ─────────────────────────────────────────────────────────")
    ok = True
    for cat_dir in sorted(DATASET_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            res_dir    = ds_dir / "resources"
            referenced: set = set()

            for split_file in ("train.jsonl", "test.jsonl"):
                jl = ds_dir / split_file
                if not jl.exists():
                    continue
                for row in _read_jsonl(jl):
                    paths = list(row.get("images") or [])
                    if row.get("video"):
                        paths.append(row["video"])
                    for p in paths:
                        abs_p = ds_dir / p
                        referenced.add(abs_p)
                        if not abs_p.exists():
                            print(f"  [MISSING] {abs_p.relative_to(REPO_ROOT)}")
                            ok = False

            corpus = ds_dir / "corpus.jsonl"
            if corpus.exists():
                for line in open(corpus):
                    for p in json.loads(line).get("corpus_images", []):
                        abs_p = ds_dir / p
                        referenced.add(abs_p)
                        if not abs_p.exists():
                            print(f"  [MISSING corpus] {abs_p.relative_to(REPO_ROOT)}")
                            ok = False

            if res_dir.exists():
                for f in list(res_dir.iterdir()):
                    if f not in referenced:
                        f.unlink()

    print("  All resource links OK." if ok else "  Some links are broken!")
    return ok


def print_summary() -> None:
    print("\n── Summary ──────────────────────────────────────────────────────────────")
    total_train = total_test = 0
    for cat_dir in sorted(DATASET_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            for split in ("train", "test"):
                jl = ds_dir / f"{split}.jsonl"
                if jl.exists():
                    n   = sum(1 for l in open(jl) if l.strip())
                    tag = f"{cat_dir.name}/{ds_dir.name}"
                    print(f"  {tag:32s} {split:5s}  {n:5d}")
                    if split == "train":
                        total_train += n
                    else:
                        total_test  += n
    print(f"\n  {'TOTAL':32s} train  {total_train:5d}")
    print(f"  {'':32s} test   {total_test:5d}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--category", metavar="CAT",
                   help="One category: Text | VL-Web | VL-Doc | Omni")
    g.add_argument("--dataset",  metavar="NAME",
                   help=f"One dataset key: {', '.join(REGISTRY)}")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--hf-token",   default=None,
                   help="HuggingFace token for gated datasets")
    args = p.parse_args()

    if args.verify_only:
        verify_all()
        print_summary()
        return

    token = args.hf_token or os.environ.get("HF_TOKEN")

    if args.dataset:
        want = {args.dataset}
    elif args.category:
        want = {k for k, v in REGISTRY.items() if v["cat"] == args.category}
    else:
        want = set(REGISTRY.keys())

    # ── Text ──────────────────────────────────────────────────────────────────
    if "AIME2024"    in want: prepare_aime2024(REGISTRY["AIME2024"])
    if "AIME2025"    in want: prepare_aime2025(REGISTRY["AIME2025"])
    if "WebWalkerQA" in want: prepare_webwalkerqa(REGISTRY["WebWalkerQA"])

    # ── VL-Web ────────────────────────────────────────────────────────────────
    for key in ("FVQA", "InfoSeek", "SimpleVQA", "LiveVQA"):
        if key in want:
            prepare_vl_web(REGISTRY[key])

    # ── VL-Doc ────────────────────────────────────────────────────────────────
    for key in ("DocBench", "MMLongBench-Doc"):
        if key in want:
            prepare_vl_doc(REGISTRY[key])

    # ── Omni ──────────────────────────────────────────────────────────────────
    if "OmniVideoBench" in want:
        prepare_omni(REGISTRY["OmniVideoBench"],
                     hf_id="NJU-LINK/OmniVideoBench", split="test",
                     fallback_id="mteb/OmniVideoBench_subset", token=token)
    if "WorldSense" in want:
        prepare_omni(REGISTRY["WorldSense"],
                     hf_id="lmms-lab/WorldSense", split="test",
                     token=token)

    verify_all()
    print_summary()
    print("\nDone.")


if __name__ == "__main__":
    main()
