"""
Configuration dataclasses for HyperSkill.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import os


# Bundled per-modality YAMLs live next to this file (hyperskill/configs/)
CONFIGS_DIR = Path(__file__).parent / "configs"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """LLM and embedding model settings (apiyi + Gemini)."""
    base_url: str = "https://api.apiyi.com/v1"
    api_key: str = ""                            # fallback to api_key_apiyi.txt or env
    model: str = "gemini-3.1-flash-lite-preview"
    embedding_model: str = "gemini-embedding-2-preview"
    embedding_dim: int = 3072
    max_tokens: int = 2048
    temperature: float = 0.7
    max_retries: int = 3
    timeout: float = 60.0


@dataclass
class HypergraphConfig:
    """Omni-modal weighted hypergraph settings."""
    working_dir: str = "expr/skill_library"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 10
    # Retrieval quality gating (applied at test/inference time)
    test_top_k: int = 3              # max skills injected during inference
    test_min_similarity: float = 0.2 # discard hits below this cosine similarity at inference
    test_min_advantage:  float = 0.0 # require learned advantage ≥ this to inject the skill
    # Skill deduplication: cosine similarity threshold for near-duplicate suppression.
    # Set to 1.0 to disable during training (avoids expensive merge LLM calls).
    skill_dedup_threshold: float = 1.0
    # Advantage update physics
    eta: float = 0.1       # driving force coefficient
    beta: float = 0.01     # KL restoring force coefficient
    eps: float = 0.2       # clipping range
    # Pruning
    prune_interval: int = 500      # prune every N updates
    prune_min_advantage: float = -0.5  # prune nodes below this weight


@dataclass
class RolloutConfig:
    """Rollout / generation settings."""
    n: int = 4                       # rollout samples per question
    max_gen_round: int = 3           # max agentic rounds per rollout
    max_prompt_length: int = 8096    # word budget
    max_new_tokens: int = 1024
    image_search_limit: int = 2
    text_search_limit: int = 3
    max_search_tokens: int = 1200    # words in a search snippet


@dataclass
class EarlyStopConfig:
    """Early-stopping criteria for the training loop.

    Stops as soon as ANY of the patience-based criteria fires (after
    `min_batches`). Set `enabled=False` (or all patiences to 0) to disable.
    """
    enabled: bool = True
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
    hop: str = "single"              # "single" | "multi"
    dataset_names: List[str] = field(default_factory=list)
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    batch_size: int = 4
    shuffle: bool = True
    max_samples: Optional[int] = None   # limit dataset size for debugging


@dataclass
class HyperSkillConfig:
    """Top-level config combining all sub-configs."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    hypergraph: HypergraphConfig = field(default_factory=HypergraphConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    data: DataConfig = field(default_factory=DataConfig)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)

    policy_model: str = "gemini-3.1-flash-lite-preview"
    mode: str = "train"              # "train" | "test"
    exp_name: str = "hyperskill_exp"
    output_dir: str = "outputs"
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "hyperskill"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _dict_to_dataclass(dc_type, d: Dict[str, Any]):
    """Recursively instantiate a dataclass from a dict."""
    if d is None:
        return dc_type()
    fields_map = {f.name: f for f in dc_type.__dataclass_fields__.values()}
    kwargs = {}
    for key, val in d.items():
        if key not in fields_map:
            continue
        f = fields_map[key]
        # check if the field type is also a dataclass
        ft = f.type if isinstance(f.type, type) else None
        if ft is not None and hasattr(ft, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[key] = _dict_to_dataclass(ft, val)
        else:
            kwargs[key] = val
    return dc_type(**kwargs)


def load_config(yaml_path: str, base: Optional["HyperSkillConfig"] = None) -> "HyperSkillConfig":
    """
    Load a YAML config file into a HyperSkillConfig dataclass.

    If `base` is provided, only fields present in the YAML override it
    (partial/overlay mode). Otherwise a fresh config is built from defaults
    plus whatever the YAML specifies.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    cfg = base if base is not None else HyperSkillConfig()

    # Sub-config sections — overlay only fields present in YAML
    for section, dc_type, attr in (
        ("llm",        LLMConfig,        "llm"),
        ("hypergraph", HypergraphConfig, "hypergraph"),
        ("rollout",    RolloutConfig,    "rollout"),
        ("data",       DataConfig,       "data"),
        ("early_stop", EarlyStopConfig,  "early_stop"),
    ):
        if section in raw:
            sub = getattr(cfg, attr)
            for key, val in raw[section].items():
                if hasattr(sub, key):
                    setattr(sub, key, val)

    # Top-level scalars
    for key in ("policy_model", "mode", "exp_name",
                "output_dir", "seed", "use_wandb", "wandb_project"):
        if key in raw:
            setattr(cfg, key, raw[key])

    # API key: YAML > env var
    if not cfg.llm.api_key:
        cfg.llm.api_key = os.environ.get("OPENAI_API_KEY", "")

    return cfg
