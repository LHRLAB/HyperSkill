# HyperSkill

---

## Install

```bash
conda create -n hyperskill python=3.13
conda activate hyperskill
pip install -r requirements.txt
sudo apt-get install ffmpeg   # for audio/video processing
```

---

## API Keys

**LLM / Embedding** — all calls go through `https://api.apiyi.com/v1`:
- Generation: `gemini-3.1-flash-lite-preview`
- Embedding:  `gemini-embedding-2-preview` (dim=3072)

```bash
echo "sk-your-apiyi-key" > api_key_apiyi.txt
```

---

## Evaluation

Three commands reproduce all reported numbers — each runs all 11 datasets in
parallel (one subprocess per dataset, logs in `outputs/logs/`):

```bash
python run.py --zero-shot  # baseline: frozen LLM, no skill retrieval
python run.py --train      # self-evolve skill libraries on all 11 datasets
python run.py --eval       # evaluate skill-augmented policy on all 11 datasets
```