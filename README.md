# HyperSkill

Official code for **HyperSkill: Training-Free Omnimodal GRPO via Hypergraph-Indexed Skill-Library Evolution**.

## 1. Environment Setup

```bash
conda create -n hyperskill python=3.13
conda activate hyperskill

pip install -r requirements.txt
sudo apt-get install ffmpeg   # optional: fallback audio extraction for WorldSense if a .wav is missing
```

All LLM and embedding calls go through OpenRouter (`base_url` in `hyperskill/configs/*.yaml`, default `https://openrouter.ai/api/v1`). Create an OpenRouter API key.

- Backbone: `google/gemini-3.1-flash-lite-preview`
- Embedding: `google/gemini-embedding-2-preview` (dim = 3072)

Web search (WebWalkerQA and the VL-Web datasets) is done through SerpAPI. You need a SerpAPI API key.

Put both keys in the repository root (both files are git-ignored):

```bash
echo "your_openrouter_key" > api_key_openrouter.txt   # or: export OPENROUTER_API_KEY=...
echo "your_serpapi_key" > api_key_serpapi.txt   # or: export SERPAPI_KEY=...
```

## 2. Data Preparation

We evaluate on HyperSkill-Bench, 11 public datasets in four regimes:

| Regime | Datasets |
|:---|:---|
| Text | AIME2024, AIME2025, WebWalkerQA |
| VL-Web | FVQA, InfoSeek, SimpleVQA, LiveVQA |
| VL-Doc | DocBench, MMLongBench-Doc |
| Omni | OmniVideoBench, WorldSense |

Download the processed HyperSkill-Bench from [Hugging Face](https://huggingface.co/datasets/LHRLAB/HyperSkill) into `dataset/` in the repository root:

```bash
pip install -U huggingface_hub
hf download LHRLAB/HyperSkill --repo-type dataset --local-dir dataset
```

Expected layout:

```text
dataset/
├── Text/{AIME2024,AIME2025,WebWalkerQA}/
├── VL-Web/{FVQA,InfoSeek,SimpleVQA,LiveVQA}/
├── VL-Doc/{DocBench,MMLongBench-Doc}/
└── Omni/{OmniVideoBench,WorldSense}/
    # each folder: train.jsonl, test.jsonl; VL / Omni folders also have resources/ (images / PDFs / videos / audio)
```

OmniVideoBench uses 8 video keyframes only; WorldSense uses 8 keyframes plus the audio track (`.wav`).

## 3. Implementation

```bash
python run.py --train && python run.py --eval
```

Each step runs all 11 datasets in parallel (logs in `outputs/logs/`). Results are written to `outputs/hyperskill_<dataset>/eval_results.json`.

`--eval` loads the skill library produced by `--train`.

Files written at runtime:

| Path | Content |
|:---|:---|
| `expr/skill_library/<dataset>/` | Skill library learned by `--train`, read by `--eval` |
| `outputs/` | Logs, training metrics and evaluation results |
| `cache/llm/` | Cached LLM responses (disable with `HYPERSKILL_DISABLE_LLM_CACHE=1`) |
| `cache/serpapi/` | Cached SerpAPI results (disable with `HYPERSKILL_DISABLE_SERPAPI_CACHE=1`) |
| `dataset/VL-Doc/<dataset>/.doc_search_chunks.json` | PDF text chunks for BM25, built on first use (`dataset/` must be writable) |
