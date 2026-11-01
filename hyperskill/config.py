"""
Configuration dataclasses for HyperSkill.

Every dataset is configured by the YAML of its modality (hyperskill/configs/
{text,vl,omni}.yaml). A YAML has four kinds of sections:

  llm / hypergraph / rollout / early_stop   shared by training and evaluation
  library_runs.<R1|R2|R3>                   training settings of a library run
  eval                                      evaluation defaults
  datasets.<name>                           the dataset's library run and its
                                            evaluation overrides

`load_config(yaml_path, dataset, mode)` merges them in this order:
  train: shared -> library_runs[datasets.<name>.library_run]
  test:  shared -> eval -> datasets.<name>.eval
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import yaml


# Bundled per-modality YAMLs live next to this file (hyperskill/configs/)
CONFIGS_DIR = Path(__file__).parent / "configs"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """LLM and embedding model settings (OpenRouter). The API key is read from
    the OPENROUTER_API_KEY environment variable (hyperskill/llm.py)."""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-3.1-flash-lite-preview"
    embedding_model: str = "google/gemini-embedding-2-preview"
    embedding_dim: int = 3072
    temperature: float = 0.4          # rollout temperature (evaluation: 0)
    max_retries: int = 3
    timeout: float = 60.0


@dataclass
class HypergraphConfig:
    """Weighted skill hypergraph: distillation, evolution and gated retrieval."""
    working_dir: str = "expr/skill_library"

    # -- Counterfactually verified step-level distillation (Sec. 3.1) ---------
    rho: float = 0.5                  # success / failure split of a rollout group
    cf_k: int = 2                     # K counterfactual probes per candidate on the
                                      # failed rollouts of its own question; admitted
                                      # on a strict majority over K_eff = min(K, |O-_q|)
    cf_distinct_failures: bool = True # False: K votes cycle over the own failures
                                      # (library run R1: 2 votes on the same failure)

    # -- Retrieval-feedback hypergraph evolution (Sec. 3.2) -------------------
    eta: float = 0.1                  # EMA step of the node advantage q_i
    prune_interval: int = 500         # pruning period (every prune_interval // 50 batches)
    prune_min_advantage: float = -0.5 # q_min: nodes below this advantage are pruned
    skill_dedup_threshold: float = 1.0  # cosine above which a new skill merges into an old one

    # -- Retrieval (Eq. 6) ----------------------------------------------------
    top_k: int = 3                    # N_max during training
    train_min_similarity: float = 0.2 # similarity floor during training
    rrf_k: int = 60                   # RRF constant k_rrf

    # -- Gated retrieval at evaluation (Sec. 3.3, Eq. 7) ---------------------
    test_top_k: int = 3               # N_max: capacity gate
    test_min_similarity: float = 0.55 # tau: similarity floor; below it -> zero-skill fallback
    test_min_advantage: float = -2.0  # advantage floor (-2 = none; q_i is in [-1, 1])

    # -- Injection (App. A.3 and per-dataset variants) ------------------------
    # P7 strategy-hint synthesis: when >= 1 skill passes the gates, one LLM
    # call distils them into a question-aware 2-3 sentence hint; otherwise the
    # skills are injected as a terse "name: description | ..." list.
    hint_synthesis: bool = True
    # (i) hub-corrected similarity s(q, v) - mu_v, mu_v = mean similarity of
    #     skill v to the training questions, used for ranking and for tau.
    hub_correction: bool = False
    # (ii) schema-constrained hint: P7 keeps the output schema (short final
    #      answer in <answer> tags) and, in tool regimes, the tool budget.
    schema_constrained_hint: bool = False
    # (iii) tool-neutral hint: in tool regimes, P7 never tells the agent to search.
    tool_neutral_hint: bool = False
    # (iv) answer-form line derived from the gold answers of each skill's 5
    #      nearest training questions (option letter / short phrase / free form).
    answer_form_line: bool = False
    # (v) completion clause: in tool regimes, give the best final answer after
    #     at most the tool budget.
    completion_clause: bool = False


@dataclass
class RolloutConfig:
    """Rollout / generation settings."""
    n: int = 2                       # rollouts per question during training
    max_gen_round: int = 3           # max agentic rounds per rollout
    max_prompt_length: int = 8096    # word budget
    max_new_tokens: int = 1024
    text_search_limit: int = 2       # tool calls per question


@dataclass
class EarlyStopConfig:
    """Early-stopping criteria for the training loop.

    Stops as soon as ANY of the patience-based criteria fires (after
    `min_batches`). Set `enabled=False` to disable.
    """
    enabled: bool = False
    min_batches: int = 16            # never stop before this many batches
    f1_window: int = 8               # rolling window for F1 plateau check
    f1_min_delta: float = 0.005      # required F1 improvement over the window
    f1_patience: int = 16            # consecutive non-improving windows before stopping
    skill_patience: int = 24         # batches without any new skill insertion before stopping
    novariance_patience: int = 16    # consecutive all-no-variance batches before stopping


@dataclass
class DataConfig:
    """Dataset and modality settings."""
    modality: str = "text"           # "text" | "vl" | "omni"
    dataset_names: List[str] = field(default_factory=list)


@dataclass
class HyperSkillConfig:
    """Top-level config combining all sub-configs."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    hypergraph: HypergraphConfig = field(default_factory=HypergraphConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    data: DataConfig = field(default_factory=DataConfig)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)

    mode: str = "train"              # "train" | "test"
    library_run: str = ""            # R1 | R2 | R3 (see the modality YAML)
    exp_name: str = "hyperskill_exp"
    output_dir: str = "outputs"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_SECTIONS = ("llm", "hypergraph", "rollout", "early_stop")


def _overlay(cfg: HyperSkillConfig, block: Dict[str, Any], where: str) -> None:
    """Set the fields of `block` ({section: {field: value}}) on `cfg`."""
    for section, values in (block or {}).items():
        if section not in _SECTIONS:
            raise ValueError(f"{where}: unknown section '{section}'")
        sub = getattr(cfg, section)
        for key, val in (values or {}).items():
            if not hasattr(sub, key):
                raise ValueError(f"{where}: unknown field '{section}.{key}'")
            setattr(sub, key, val)


def load_config(yaml_path: str, dataset: str, mode: str) -> HyperSkillConfig:
    """Build the config of `dataset` for `mode` ("train" | "test")."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    ds_cfg = (raw.get("datasets") or {}).get(dataset)
    if ds_cfg is None:
        raise ValueError(f"{yaml_path}: no entry for dataset '{dataset}'")

    cfg = HyperSkillConfig()
    cfg.mode = mode
    _overlay(cfg, {s: raw[s] for s in _SECTIONS if s in raw}, yaml_path)

    cfg.library_run = ds_cfg["library_run"]
    if mode == "train":
        runs = raw.get("library_runs") or {}
        if cfg.library_run not in runs:
            raise ValueError(f"{yaml_path}: unknown library run '{cfg.library_run}'")
        _overlay(cfg, runs[cfg.library_run], f"{yaml_path}: library_runs.{cfg.library_run}")
    else:
        _overlay(cfg, raw.get("eval"), f"{yaml_path}: eval")
        _overlay(cfg, ds_cfg.get("eval"), f"{yaml_path}: datasets.{dataset}.eval")
    return cfg
