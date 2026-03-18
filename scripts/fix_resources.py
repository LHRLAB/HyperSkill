#!/usr/bin/env python3
"""
Fix Omni and VL-Doc datasets:
  - OmniVideoBench : add separate .wav audio from HF Arrow cache
  - WorldSense     : copy videos from HyperMAS, extract audio with ffmpeg
  - DocBench       : put PDFs in resources/, update resource_paths
  - MMLongBench-Doc: put PDFs in resources/, update resource_paths
  - Clean up       : remove corpus.jsonl (replaced by PDFs)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT    = Path(__file__).parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"
HYPMAS_WS    = Path("/mnt/hdd2/home/luohaoran/github/HyperMAS/dataset/Omni/WorldSense")


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows → {path.relative_to(REPO_ROOT)}")


def _extract_audio_ffmpeg(mp4: Path, wav: Path) -> bool:
    """Extract audio stream from mp4 to wav using ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", str(wav)],
        capture_output=True,
    )
    return wav.exists() and wav.stat().st_size > 0


# ── OmniVideoBench : add audio from HF Arrow cache ───────────────────────────

def fix_omnividobench() -> None:
    print("\n── OmniVideoBench: extracting audio from Arrow cache")
    import pyarrow as pa
    import pyarrow.ipc as ipc

    ds_dir  = DATASET_ROOT / "Omni" / "OmniVideoBench"
    res_dir = ds_dir / "resources"

    # Load all arrow files → map index → audio bytes
    cache_root = Path.home() / ".cache/huggingface/datasets/mteb___omni_video_bench_subset"
    arrow_files = sorted(cache_root.rglob("*-test-*.arrow"))

    all_items = []
    for af in arrow_files:
        try:
            with pa.memory_map(str(af), "r") as src:
                table = ipc.open_stream(src).read_all()
            for i in range(len(table)):
                all_items.append({col: table.column(col)[i].as_py()
                                   for col in table.schema.names})
        except Exception:
            pass

    import random
    random.Random(42).shuffle(all_items)

    # build map: video_id → audio bytes
    audio_map: Dict[str, bytes] = {}
    for it in all_items:
        vid_id = it.get("video_id", "")
        aud    = it.get("audio") or {}
        if isinstance(aud, dict):
            b = aud.get("bytes") or b""
        else:
            b = aud or b""
        if vid_id and b:
            audio_map[vid_id] = b

    # update each split
    for split in ("train", "test"):
        rows = _read_jsonl(ds_dir / f"{split}.jsonl")
        changed = 0
        for r in rows:
            # video filename without ext = index (e.g. "resources/00000.mp4")
            vp     = r.get("video") or ""
            stem   = Path(vp).stem  # "00000"
            wav_fn = f"{stem}.wav"
            wav_p  = res_dir / wav_fn

            # find matching arrow item by index
            idx  = int(stem)
            item = all_items[idx] if idx < len(all_items) else None
            if item is None:
                continue
            aud = item.get("audio") or {}
            b   = aud.get("bytes") if isinstance(aud, dict) else (aud or b"")
            if b and isinstance(b, (bytes, bytearray)):
                wav_p.write_bytes(b)
                r["audio"] = f"resources/{wav_fn}"
                changed += 1
        _write_jsonl(ds_dir / f"{split}.jsonl", rows)
        print(f"    {split}: audio added for {changed} rows")


# ── WorldSense : copy videos + extract audio ─────────────────────────────────

def fix_worldsense() -> None:
    print("\n── WorldSense: rebuilding from HyperMAS data")
    ds_dir  = DATASET_ROOT / "Omni" / "WorldSense"
    res_dir = ds_dir / "resources"
    res_dir.mkdir(exist_ok=True)

    split_map = {
        "train": HYPMAS_WS / "WorldSense_train.jsonl",
        "test":  HYPMAS_WS / "WorldSense_test.jsonl",
    }

    for split, src_path in split_map.items():
        src_rows = _read_jsonl(src_path)
        out_rows = []
        no_vid = missing_vid = ok_vid = 0

        for i, r in enumerate(src_rows):
            q = r.get("question", "")
            # candidates: stored as string repr of list in HyperMAS
            cands_raw = r.get("candidates", [])
            if isinstance(cands_raw, str):
                try:
                    import ast
                    cands_raw = ast.literal_eval(cands_raw)
                except Exception:
                    cands_raw = []
            # append options to question
            if cands_raw:
                q += "\n" + "\n".join(f"({chr(65+j)}) {o}" for j, o in enumerate(cands_raw))

            ans = r.get("answer", "")
            src_vid = HYPMAS_WS / r.get("video_path", "")
            vid_fn  = f"{split}_{i:05d}.mp4"
            wav_fn  = f"{split}_{i:05d}.wav"
            out_vid = res_dir / vid_fn
            out_wav = res_dir / wav_fn

            video_rel = audio_rel = None

            if src_vid.exists():
                shutil.copy2(src_vid, out_vid)
                video_rel = f"resources/{vid_fn}"
                ok_vid += 1
                # extract audio from video
                if _extract_audio_ffmpeg(out_vid, out_wav):
                    audio_rel = f"resources/{wav_fn}"
            else:
                missing_vid += 1

            out_rows.append({
                "question":    q,
                "gold_answers": [ans],
                "data_source":  "WorldSense",
                "task_type":    "mmreasoning",
                "modality":     "omni",
                "images":       [],
                "video":        video_rel,
                "audio":        audio_rel,
            })

        _write_jsonl(ds_dir / f"{split}.jsonl", out_rows)
        print(f"    {split}: {ok_vid} videos copied, {missing_vid} missing")

    # remove old empty resources
    for f in res_dir.iterdir():
        if f.suffix in (".mp4", ".wav"):
            # already written above; skip stale files not referenced
            pass


# ── DocBench : extract PDFs from zip ─────────────────────────────────────────

def fix_docbench() -> None:
    print("\n── DocBench: extracting PDFs from zip")
    ds_dir  = DATASET_ROOT / "VL-Doc" / "DocBench"
    res_dir = ds_dir / "resources"
    res_dir.mkdir(exist_ok=True)

    zpath = next(Path.home().glob(
        ".cache/huggingface/hub/datasets--icy123123--DocBench/**/*.zip"))
    with zipfile.ZipFile(zpath) as z:
        name2member = {Path(m).name: m for m in z.namelist() if m.endswith(".pdf")}

        for split in ("train", "test"):
            rows = _read_jsonl(ds_dir / f"{split}.jsonl")
            changed = 0
            for r in rows:
                m = re.search(r"from (.+?\.pdf)", r["question"])
                if not m:
                    continue
                pdf_name = m.group(1)
                if pdf_name not in name2member:
                    continue
                dest = res_dir / pdf_name
                if not dest.exists():
                    data = z.read(name2member[pdf_name])
                    dest.write_bytes(data)
                r["images"] = [f"resources/{pdf_name}"]
                changed += 1
            _write_jsonl(ds_dir / f"{split}.jsonl", rows)
            print(f"    {split}: PDFs linked for {changed}/{len(rows)} rows")

    # remove corpus.jsonl
    corpus = ds_dir / "corpus.jsonl"
    if corpus.exists():
        corpus.unlink()
        print("    removed corpus.jsonl")


# ── MMLongBench-Doc : download PDFs from HF ──────────────────────────────────

def fix_mmlongbench() -> None:
    print("\n── MMLongBench-Doc: downloading PDFs from HuggingFace")
    ds_dir  = DATASET_ROOT / "VL-Doc" / "MMLongBench-Doc"
    res_dir = ds_dir / "resources"
    res_dir.mkdir(exist_ok=True)

    from huggingface_hub import hf_hub_download

    # collect needed PDF names
    needed: set = set()
    for split in ("train", "test"):
        for r in _read_jsonl(ds_dir / f"{split}.jsonl"):
            m = re.search(r"from (.+?\.pdf)", r["question"])
            if m:
                needed.add(m.group(1))

    # download each PDF
    downloaded = 0
    for pdf_name in sorted(needed):
        dest = res_dir / pdf_name
        if dest.exists():
            downloaded += 1
            continue
        try:
            tmp = hf_hub_download(
                "yubo2333/MMLongBench-Doc",
                f"documents/{pdf_name}",
                repo_type="dataset",
            )
            shutil.copy2(tmp, dest)
            downloaded += 1
        except Exception as e:
            print(f"    [skip] {pdf_name}: {e}")

    print(f"    downloaded {downloaded}/{len(needed)} PDFs")

    # update JSONL
    for split in ("train", "test"):
        rows = _read_jsonl(ds_dir / f"{split}.jsonl")
        changed = 0
        for r in rows:
            m = re.search(r"from (.+?\.pdf)", r["question"])
            if not m:
                continue
            pdf_name = m.group(1)
            if (res_dir / pdf_name).exists():
                r["images"] = [f"resources/{pdf_name}"]
                changed += 1
        _write_jsonl(ds_dir / f"{split}.jsonl", rows)
        print(f"    {split}: PDFs linked for {changed}/{len(rows)} rows")

    corpus = ds_dir / "corpus.jsonl"
    if corpus.exists():
        corpus.unlink()
        print("    removed corpus.jsonl")


# ── clean resources : remove unreferenced files ───────────────────────────────

def clean_resources() -> None:
    print("\n── Cleaning unreferenced resource files")
    for cat_dir in sorted(DATASET_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            res_dir = ds_dir / "resources"
            if not res_dir.exists():
                continue
            referenced: set = set()
            for split in ("train.jsonl", "test.jsonl"):
                jl = ds_dir / split
                if not jl.exists():
                    continue
                for row in _read_jsonl(jl):
                    for p in list(row.get("images") or []):
                        referenced.add(res_dir / Path(p).name)
                    if row.get("video"):
                        referenced.add(res_dir / Path(row["video"]).name)
                    if row.get("audio"):
                        referenced.add(res_dir / Path(row["audio"]).name)
            removed = 0
            for f in list(res_dir.iterdir()):
                if f not in referenced:
                    f.unlink()
                    removed += 1
            if removed:
                print(f"  {cat_dir.name}/{ds_dir.name}: removed {removed} unreferenced files")


# ── final check ───────────────────────────────────────────────────────────────

def final_check() -> None:
    print("\n── Final check")
    for cat_dir in sorted(DATASET_ROOT.iterdir()):
        if not cat_dir.is_dir():
            continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            # check for stray files (only allow train.jsonl, test.jsonl, resources/)
            for f in ds_dir.iterdir():
                if f.name not in ("train.jsonl", "test.jsonl", "resources"):
                    print(f"  [STRAY] {f.relative_to(REPO_ROOT)}")
            # spot check
            for split in ("train", "test"):
                jl = ds_dir / f"{split}.jsonl"
                if not jl.exists():
                    print(f"  [MISSING] {jl.relative_to(REPO_ROOT)}")
                    continue
                rows = _read_jsonl(jl)
                empty_q = sum(1 for r in rows if not r.get("question", "").strip())
                empty_a = sum(1 for r in rows if not r.get("gold_answers"))
                miss = 0
                for r in rows:
                    for p in list(r.get("images") or []) + \
                              ([r["video"]] if r.get("video") else []) + \
                              ([r["audio"]] if r.get("audio") else []):
                        if not (ds_dir / p).exists():
                            miss += 1
                tag = f"{cat_dir.name}/{ds_dir.name}/{split}"
                issues = ""
                if empty_q: issues += f" empty_q={empty_q}"
                if empty_a: issues += f" empty_a={empty_a}"
                if miss:    issues += f" miss={miss}"
                status = "OK  " if not issues else "WARN"
                print(f"  [{status}] {tag:45s} {len(rows):5d} rows{issues}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["omni", "worldsense", "docbench",
                                      "mmlongbench", "clean", "check", "all"],
                   default="all")
    args = p.parse_args()

    if args.step in ("omni",       "all"): fix_omnividobench()
    if args.step in ("worldsense", "all"): fix_worldsense()
    if args.step in ("docbench",   "all"): fix_docbench()
    if args.step in ("mmlongbench","all"): fix_mmlongbench()
    if args.step in ("clean",      "all"): clean_resources()
    if args.step in ("check",      "all"): final_check()
