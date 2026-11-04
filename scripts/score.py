"""Score evaluation results with both scorers ($0, no API call).

  hyperskill/grpo/rewards_cjk.py  CJK-aware token F1 (CJK characters are single tokens); used for
                                  the main table
  hyperskill/grpo/rewards.py      official scorer (the one the harness writes into eval_results.json)

The two differ only on answers in Chinese / Japanese / Korean (in the main table: WebWalkerQA).

    python scripts/score.py --tag hs                # outputs/hs_hyperskill_<dataset>/eval_results.json, all found
    python scripts/score.py path/to/eval_results.json [...]

A question whose rollout raised an error keeps the stored score (0).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hyperskill.datasets.loaders import DATASET_KEYS  # noqa: E402
from hyperskill.grpo import rewards, rewards_cjk  # noqa: E402

EXPECTED_CJK = {"AIME2024": 40.0, "AIME2025": 53.3, "WebWalkerQA": 21.1, "FVQA": 42.6, "InfoSeek": 44.8,
                "SimpleVQA": 40.6, "LiveVQA": 64.5, "DocBench": 39.2, "MMLongBench-Doc": 32.1,
                "OmniVideoBench": 31.5, "WorldSense": 51.6}


def score_file(path: Path) -> dict:
    r = json.loads(path.read_text())
    off, cjk = [], []
    for q in r["per_question"]:
        gold, pred = q.get("gold_all") or q["gold"], q.get("response_full")
        if pred is None or q.get("error"):
            off.append(q["f1"]); cjk.append(q["f1"]); continue
        off.append(rewards.compute_f1_reward(pred, gold))
        cjk.append(rewards_cjk.compute_f1_reward(pred, gold))
    n = len(off)
    return {"dataset": r.get("dataset"), "n": n, "n_errors": r.get("n_errors"),
            "f1_official": 100 * sum(off) / n, "f1_cjk": 100 * sum(cjk) / n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--tag", default=None, help="score outputs/<tag>_hyperskill_<dataset>/eval_results.json")
    ap.add_argument("--json", type=Path, default=None, help="also write the scores to this file")
    a = ap.parse_args()
    files = list(a.files)
    if a.tag is not None:
        prefix = f"{a.tag}_" if a.tag else ""
        files += [f for ds in DATASET_KEYS
                  if (f := ROOT / "outputs" / f"{prefix}hyperskill_{ds.lower()}" / "eval_results.json").exists()]
    if not files:
        ap.error("no eval_results.json given or found")
    rows = [score_file(f) for f in files]
    print("| Dataset | n | errors | F1 (CJK scorer) | F1 (official) | paper (CJK) |")
    print("|:--|--:|--:|--:|--:|--:|")
    for s in rows:
        print(f"| {s['dataset']} | {s['n']} | {s['n_errors']} | {s['f1_cjk']:.2f} | {s['f1_official']:.2f} | "
              f"{EXPECTED_CJK.get(s['dataset'], '')} |")
    if len(rows) > 1:
        m = len(rows)
        print(f"| **Average ({m})** | | | **{sum(s['f1_cjk'] for s in rows) / m:.2f}** | "
              f"**{sum(s['f1_official'] for s in rows) / m:.2f}** | {'41.9' if m == 11 else ''} |")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
