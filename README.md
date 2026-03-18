# HyperSkill

Self-Evolving Omni-Modal Weighted Hypergraph Skill Library for Training-Free GRPO.

> Method design: see [Paper.md](Paper.md)

---

## Installation

```bash
conda create -n hyperskill python=3.13
conda activate hyperskill
pip install -r requirements.txt
```

---

## Data Preparation

Download the dataset from HuggingFace:

```bash
huggingface-cli download <HF_REPO_ID> --repo-type dataset --local-dir dataset
```

### Dataset layout

```
dataset/
  Text/
    AIME2024/          train.jsonl (1024)   test.jsonl (30)
    AIME2025/          train.jsonl (1024)   test.jsonl (30)
    WebWalkerQA/       train.jsonl (616)    test.jsonl (64)
  VL-Web/
    FVQA/              train.jsonl (1024)   test.jsonl (64)    resources/
    InfoSeek/          train.jsonl (1024)   test.jsonl (64)    resources/
    SimpleVQA/         train.jsonl (949)    test.jsonl (64)    resources/
    LiveVQA/           train.jsonl (1024)   test.jsonl (64)    resources/
  VL-Doc/
    DocBench/          train.jsonl (472)    test.jsonl (64)    resources/
    MMLongBench-Doc/   train.jsonl (501)    test.jsonl (64)    resources/
  Omni/
    OmniVideoBench/    train.jsonl (445)    test.jsonl (64)    resources/
    WorldSense/        train.jsonl (1024)   test.jsonl (64)    resources/
```

JSONL schema: `question, gold_answers, data_source, task_type, modality, images, video, audio`

- `images`: list of relative paths to image/PDF files in `resources/`
- `video`: relative path to `.mp4` in `resources/`, or `null`
- `audio`: relative path to `.wav` in `resources/`, or `null`

---

## API Keys

**LLM / Embedding** — all calls go through `https://api.apiyi.com/v1`:
- Generation: `gemini-3.1-flash-lite-preview`
- Embedding:  `gemini-embedding-2-preview` (dim=3072)

```bash
echo "sk-your-apiyi-key" > api_key_apiyi.txt
```

**SerpAPI** — required for web search tasks (`FVQA`, `InfoSeek`, `SimpleVQA`, `LiveVQA`, `WebWalkerQA`).
Get a key at [serpapi.com](https://serpapi.com):

```bash
echo "your-serpapi-key" > api_key_serpapi.txt
```

Without a SerpAPI key, web search falls back to skill library retrieval.

---

## Training Monitor (optional)

Start the real-time dashboard before training, then open `http://localhost:8888`:

```bash
python scripts/monitor.py
# Custom port:
python scripts/monitor.py --port 8080
```

Displays per-dataset F1 curves, skill node growth, elapsed time, and a live event log.

---

## Step 1 — Self-Evolution Training

The skill library is built and saved to `expr/skill_library/<dataset>/`.

```bash
# Train all datasets (default: reset library + 1 epoch)
python scripts/evolve_skills.py --all

# Train a single dataset
python scripts/evolve_skills.py --dataset dataset/Text/AIME2024
python scripts/evolve_skills.py --dataset dataset/Text/WebWalkerQA
python scripts/evolve_skills.py --dataset dataset/VL-Doc/DocBench
python scripts/evolve_skills.py --dataset dataset/VL-Web/FVQA
python scripts/evolve_skills.py --dataset dataset/Omni/OmniVideoBench

# Multiple epochs
python scripts/evolve_skills.py --all --epochs 3

# Resume on existing library (skip reset)
python scripts/evolve_skills.py --all --resume
python scripts/evolve_skills.py --all --resume --epochs 2

# Delete skill libraries without training
python scripts/evolve_skills.py --all --delete-only

# Quick smoke test (50 samples)
python scripts/evolve_skills.py --dataset dataset/Text/AIME2024 --max-samples 50
```

Checkpoints are saved to `outputs/hyperskill_<dataset>/epoch_N/metrics.json`.
Training log (read by the monitor) is written to `outputs/train_log.jsonl`.

---

## Step 2 — Evaluation

```bash
# Evaluate all datasets
python scripts/evaluate.py --all

# Evaluate a single dataset
python scripts/evaluate.py --dataset dataset/Text/AIME2024

# Zero-shot baseline (no skill retrieval)
python scripts/evaluate.py --all --no-skills

# Quick eval on a subset
python scripts/evaluate.py --dataset dataset/Omni/OmniVideoBench --max-samples 50
```

Results are written to `outputs/eval_results.json`.

---

## Interactive Agent Demo

```python
import asyncio
from hyperskill.config import HyperSkillConfig
from hyperskill.agents.hyperskill_agent import HyperSkillAgent

async def demo():
    cfg = HyperSkillConfig()
    cfg.hypergraph.working_dir = "expr/skill_library/AIME2024"
    agent = HyperSkillAgent(cfg)

    result = await agent.answer(
        question="Which city is the capital of the country that borders France to the northeast?",
        task_type="websearch",
        verbose=True,
    )
    print("Answer:", result["answer"])
    print("Skills used:", [s.name for s in result["skills_used"]])

asyncio.run(demo())
```

For VL / Omni tasks, pass image bytes:

```python
with open("scan.jpg", "rb") as f:
    img_bytes = f.read()

result = await agent.answer(
    question="What disease is shown in this medical scan?",
    task_type="mmreasoning",
    images=[img_bytes],
)
print(result["answer"])
```

---

## Configuration Reference

| Field | Default | Description |
|-------|---------|-------------|
| `llm.model` | `gemini-3.1-flash-lite-preview` | Generation model |
| `llm.embedding_model` | `gemini-embedding-2-preview` | Embedding model (dim=3072) |
| `hypergraph.working_dir` | `expr/skill_library/…` | Skill library path |
| `hypergraph.top_k` | `10` | Skills retrieved per query |
| `hypergraph.eta` | `0.1` | Driving force coefficient |
| `hypergraph.beta` | `0.01` | KL restoring force coefficient |
| `hypergraph.skill_dedup_threshold` | `0.85` | Cosine similarity threshold for merging skills |
| `hypergraph.prune_min_advantage` | `-0.5` | Prune skills below this advantage weight |
| `rollout.n` | `5` | Rollout samples per question |
| `rollout.max_gen_round` | `10` | Max agentic rounds per rollout |
| `data.batch_size` | `8` | Questions per training batch |
