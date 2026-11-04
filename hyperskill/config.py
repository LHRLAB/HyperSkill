"""
Configuration dataclasses for HyperSkill.

Every dataset is configured by the YAML of its modality (hyperskill/configs/
{text,vl,omni}.yaml). A YAML has four kinds of sections:

  llm / hypergraph / rollout / early_stop   shared by training and evaluation
  library_runs.<R1|R2|R3>                   training settings of a library run
  eval                                      evaluation defaults
  datasets.<name>                           the dataset's library run and its
                                            shared harness settings

`load_config(yaml_path, dataset, mode)` merges them in this order:
  train: shared -> library_runs[datasets.<name>.library_run]
  test:  shared -> eval -> datasets.<name>.eval (if any)

`override` (one dataset's entry of a configs JSON, by default
configs/final_configs.json; see run.py --configs) can replace the dataset's
entry: `library_run` replaces datasets.<name>.library_run, `train` is overlaid
after the library run, and `eval`, when present, replaces datasets.<name>.eval.
Last, datasets.<name>.harness (shared harness settings such as the number of
video frames) is applied in both modes.
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
    model: str = "google/gemini-2.5-flash-lite"
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
    # (iii) answer-form line derived from the gold answers of each skill's 5
    #       nearest training questions (option letter / short phrase / free form).
    answer_form_line: bool = False
    # (iv) protocol skill (hyperskill/strategies/protocol.py): the output-protocol
    #      failure types seen in >= protocol_min_count training rollouts (malformed
    #      tool call, tool call after the budget, tool call without tools, no
    #      <answer>) become one always-injected skill, re-injected after each tool
    #      turn with the remaining budget. Training: injected once a type is seen.
    protocol_skill: bool = False
    protocol_min_count: int = 2
    # "short": fixed short-answer line; "data": answer line derived from the
    # training gold answers (option letter / short phrase / one sentence, plus
    # the exact wording of a gold string that covers >= 10% of them)
    protocol_answer_form: str = "short"

    # -- Agent strategies (hyperskill/strategies/; all off by default) --------
    # Prompt and tool calls (strategies/prompting.py):
    #   the search format names the real tool tag with one concrete call (no
    #   TOOL_TAG placeholder); no search format when no tool is offered.
    concrete_tool_prompt: bool = False
    #   one final-answer turn when a rollout would end without a closed <answer>.
    final_answer_turn: bool = False
    #   answer form after the question: math -> integer in <answer> (no \boxed{});
    #   multiple choice -> option letter in <answer>.
    answer_suffix: bool = False
    #   the tool-call parser also accepts the TOOL_TAG placeholder forms.
    tool_parse_lenient: bool = False
    #   doc_search on a PDF without indexed text: explicit message, no search.
    doc_own_only: bool = False
    # Evaluation only:
    #   self-consistency (strategies/voting.py): vote_k rollouts sampled at
    #   vote_temperature besides the greedy one, majority vote on the final answer,
    #   tie -> greedy; vote_skip_refusals: refusal answers do not vote.
    vote_k: int = 0
    vote_temperature: float = 0.7
    vote_skip_refusals: bool = False
    #   doc QA (strategies/locate_pages.py): text of the top-n BM25 pages of the
    #   question's own PDF after the question (0 = off).
    locate_pages: int = 0
    #   web QA (strategies/presearch.py): the question itself is sent as a first
    #   knowledge-search query; results after the question.
    question_presearch: bool = False
    #   omni multiple choice (strategies/video.py): video length and the time of
    #   every keyframe after the question / per-option evidence check line.
    time_grounding: bool = False
    option_check: bool = False


@dataclass
class RolloutConfig:
    """Rollout / generation settings."""
    n: int = 2                       # rollouts per question during training
    max_gen_round: int = 3           # max agentic rounds per rollout
    max_prompt_length: int = 8096    # word budget
    max_new_tokens: int = 1024
    text_search_limit: int = 2       # tool calls per question
    max_frames: int = 8              # video keyframes per question


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
    train_split: str = "train128"    # questions the library is trained on; also used by
                                     # the hub correction and the answer-form line


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


def load_config(yaml_path: str, dataset: str, mode: str,
                override: Dict[str, Any] | None = None) -> HyperSkillConfig:
    """Build the config of `dataset` for `mode` ("train" | "test")."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    ds_cfg = (raw.get("datasets") or {}).get(dataset)
    if ds_cfg is None:
        raise ValueError(f"{yaml_path}: no entry for dataset '{dataset}'")
    override = override or {}
    unknown = set(override) - {"library_run", "train", "eval", "seed", "note"}
    if unknown:
        raise ValueError(f"configs entry of {dataset}: unknown keys {sorted(unknown)}")
    ds_cfg = dict(ds_cfg)
    if "library_run" in override:
        ds_cfg["library_run"] = override["library_run"]
    if "eval" in override:
        ds_cfg["eval"] = override["eval"]

    cfg = HyperSkillConfig()
    cfg.mode = mode
    _overlay(cfg, {s: raw[s] for s in _SECTIONS if s in raw}, yaml_path)

    cfg.library_run = ds_cfg["library_run"]
    if mode == "train":
        runs = raw.get("library_runs") or {}
        if cfg.library_run not in runs:
            raise ValueError(f"{yaml_path}: unknown library run '{cfg.library_run}'")
        _overlay(cfg, runs[cfg.library_run], f"{yaml_path}: library_runs.{cfg.library_run}")
        _overlay(cfg, override.get("train"), f"configs entry of {dataset}: train")
    else:
        _overlay(cfg, raw.get("eval"), f"{yaml_path}: eval")
        _overlay(cfg, ds_cfg.get("eval"), f"{yaml_path}: datasets.{dataset}.eval")
    # shared harness settings of the dataset (applied last, so a configs entry
    # cannot replace them)
    _overlay(cfg, ds_cfg.get("harness"), f"{yaml_path}: datasets.{dataset}.harness")
    return cfg
