# HyperSkill

HyperSkill: Training-Free Omnimodal GRPO via Hypergraph-Indexed Skill-Library Evolution

## Overview

![](./assets/overview.png)

## Quick Start

### 1. Environment Setup

```bash
conda create -n hyperskill python=3.13 -y
conda activate hyperskill

pip install -r requirements.txt
```

All model calls go through [OpenRouter](https://openrouter.ai/): the backbone `google/gemini-3.1-flash-lite-preview`, the embedding model `google/gemini-embedding-2-preview`, and `google/gemini-3.8-flash` for the `knowledge_search` tool (WebWalkerQA and VL-Web). Create an OpenRouter API key and export it:

```bash
export OPENROUTER_API_KEY="your_openrouter_key_here"
```

### 2. Data Preparation

We use HyperSkill-Bench, 11 datasets in four regimes:

| Regime | Datasets |
|:---|:---|
| Text | AIME2024, AIME2025, WebWalkerQA |
| VL-Web | FVQA, InfoSeek, SimpleVQA, LiveVQA |
| VL-Doc | DocBench, MMLongBench-Doc |
| Omni | OmniVideoBench, WorldSense |

Download the data from [Hugging Face](https://huggingface.co/datasets/LHRLAB/HyperSkill) into `dataset/`:

```bash
hf download LHRLAB/HyperSkill --repo-type dataset --local-dir dataset
```

```text
dataset/
├── Text/{AIME2024,AIME2025,WebWalkerQA}/
├── VL-Web/{FVQA,InfoSeek,SimpleVQA,LiveVQA}/
├── VL-Doc/{DocBench,MMLongBench-Doc}/
└── Omni/{OmniVideoBench,WorldSense}/
```

Each dataset directory holds `train.jsonl` (128 training questions), `test.jsonl` (30 test questions for AIME2024/2025, 64 for the others) and `resources/` (the images, PDFs, videos and audio they reference).

### 3. Implementation

```bash
python run.py --train
python run.py --eval
```

`--train` evolves the skill library of each dataset (1 epoch over the 128 training questions, batch size 4, 2 rollouts per question) and writes it to `expr/skill_library/<dataset>/`. `--eval` answers the test questions with the learned library (1 rollout, temperature 0) and writes token-level F1 to `outputs/hyperskill_<dataset>/eval_results.json`. Both run the 11 datasets in parallel; add `--dataset <name>` to run one, e.g. `python run.py --train --dataset AIME2024`. Per-dataset settings are in [`hyperskill/configs/`](hyperskill/configs/).

## BibTeX

If you find this work is helpful for your research, please cite:

```bibtex
@inproceedings{luo2026hyperskill,
 author = {Luo, Haoran and Lin, Shangkai and Wu, Jinyang and Zhou, Xinliang and Luu, Anh Tuan},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {HyperSkill: Training-Free Omnimodal GRPO via Hypergraph-Indexed Skill-Library Evolution},
 year = {2026}
}
```

For further questions, please contact: haoran.luo@ntu.edu.sg.

## Acknowledgment

This repo benefits from [Training-Free GRPO](https://github.com/TencentCloudADP/youtu-agent), [HyperGraphRAG](https://github.com/LHRLAB/HyperGraphRAG), and [Graph-R1](https://github.com/LHRLAB/Graph-R1). Thanks for their wonderful works.
