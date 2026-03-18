"""
Reward functions for Training-Free GRPO in HyperSkill.

Three reward signals:
  1. F1 reward       — token-level F1 between predicted and ground-truth answers
  2. Format reward   — structural validity of the response format
  3. Search reward   — bonus for useful (non-repetitive) search calls
"""

from __future__ import annotations
import re
import string
from collections import Counter
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def collapse_spaces(text: str) -> str:
        return " ".join(text.split())

    def remove_punct(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return collapse_spaces(remove_articles(remove_punct(s.lower())))


def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and a single gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1_reward(
    prediction: str,
    gold_answers: Union[str, List[str]],
) -> float:
    """
    Compute F1 reward: max F1 over all gold answer alternatives.
    Gold answers may be a string or a list of strings.
    """
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    # Extract predicted answer from <answer>...</answer> tag
    answer_match = re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        pred_text = answer_match[-1].strip()
    else:
        pred_text = prediction.strip()

    return max(token_f1(pred_text, gold) for gold in gold_answers)


# ---------------------------------------------------------------------------
# Format rewards
# ---------------------------------------------------------------------------

# Allowed response patterns (regex)
_DIRECT_ANSWER_RE = re.compile(
    r"^<reason>.*?</reason>.*?<answer>.*?</answer>$", re.DOTALL
)
_TEXT_SEARCH_RE = re.compile(
    r"^<reason>.*?</reason>.*?<text_search>.*?</text_search>$", re.DOTALL
)
_IMAGE_SEARCH_RE = re.compile(
    r"^<reason>.*?</reason>.*?<search><img></search>$", re.DOTALL
)


def _is_valid_direct_answer(response: str) -> bool:
    if not _DIRECT_ANSWER_RE.match(response):
        return False
    if response.count("<reason>") != 1 or response.count("</reason>") != 1:
        return False
    if response.count("<answer>") != 1 or response.count("</answer>") != 1:
        return False
    if "<search><img></search>" in response:
        return False
    if "<text_search>" in response or "</text_search>" in response:
        return False
    return True


def _is_valid_text_search(response: str) -> bool:
    if not _TEXT_SEARCH_RE.match(response):
        return False
    if response.count("<reason>") != 1 or response.count("</reason>") != 1:
        return False
    if response.count("<text_search>") != 1 or response.count("</text_search>") != 1:
        return False
    if "<answer>" in response or "</answer>" in response:
        return False
    if "<search><img></search>" in response:
        return False
    return True


def _is_valid_image_search(response: str) -> bool:
    if not _IMAGE_SEARCH_RE.match(response):
        return False
    if response.count("<reason>") != 1 or response.count("</reason>") != 1:
        return False
    if response.count("<search><img></search>") != 1:
        return False
    if "<answer>" in response or "</answer>" in response:
        return False
    if "<text_search>" in response or "</text_search>" in response:
        return False
    return True


def compute_format_reward(
    response_list: List[str],
    task_type: str = "mmreasoning",
) -> float:
    """
    Check if a multi-turn response trajectory follows valid formats throughout.

    For mmreasoning: only direct answers allowed.
    For websearch: search actions + final direct answer allowed.

    Returns a score in {0.0, 0.5, 1.0}.
    """
    if not response_list:
        return 0.0

    # For reasoning tasks: every intermediate step must be direct answer
    if task_type == "mmreasoning":
        # All intermediate + final responses should be direct answers
        for resp in response_list:
            resp = resp.strip()
            if not _is_valid_direct_answer(resp):
                # Partial credit: at least has the right tags
                if "<reason>" in resp and "<answer>" in resp:
                    return 0.5
                return 0.0
        return 1.0

    # For search tasks: intermediates can be searches, last must be direct answer
    for i, resp in enumerate(response_list):
        resp = resp.strip()
        is_last = (i == len(response_list) - 1)
        if is_last:
            if not _is_valid_direct_answer(resp):
                if "<reason>" in resp and "<answer>" in resp:
                    return 0.5
                return 0.0
        else:
            if not (_is_valid_text_search(resp) or _is_valid_image_search(resp) or _is_valid_direct_answer(resp)):
                if "<reason>" in resp:
                    return 0.5
                return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Search utility reward
# ---------------------------------------------------------------------------

def compute_search_reward(response_list: List[str]) -> float:
    """
    Reward for useful (non-repetitive, relevant) search calls.
    Penalizes repeated identical queries.
    """
    seen_queries: set = set()
    score = 0.0
    for resp in response_list:
        text_searches = re.findall(r"<text_search>(.*?)</text_search>", resp)
        for q in text_searches:
            q = q.strip()
            if q in seen_queries:
                score -= 0.1  # penalty for repetition
            else:
                score += 0.05  # small bonus for unique searches
                seen_queries.add(q)
    return max(0.0, min(score, 0.3))  # cap bonus


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def compute_combined_reward(
    prediction: str,
    gold_answers: Union[str, List[str]],
    response_list: List[str],
    task_type: str = "mmreasoning",
    f1_weight: float = 0.8,
    format_weight: float = 0.1,
    search_weight: float = 0.1,
) -> float:
    """
    Combined reward = f1_weight * F1 + format_weight * Format + search_weight * Search
    """
    f1 = compute_f1_reward(prediction, gold_answers)
    fmt = compute_format_reward(response_list, task_type)
    if task_type == "websearch":
        srch = compute_search_reward(response_list)
    else:
        srch = 0.0

    return f1_weight * f1 + format_weight * fmt + search_weight * srch


# ---------------------------------------------------------------------------
# GRPO advantage computation
# ---------------------------------------------------------------------------

def compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """
    Compute GRPO advantages: normalize rewards within a group.
    advantage_i = (r_i - mean(r)) / (std(r) + 1e-8)
    Then apply tanh squashing to keep in [-1, 1].
    """
    import numpy as np
    if not rewards:
        return []
    r = np.array(rewards, dtype=np.float64)
    mean_r = r.mean()
    std_r = r.std() + 1e-8
    normalized = (r - mean_r) / std_r
    return [float(np.tanh(x)) for x in normalized]


def compute_grpo_advantages_global(
    rewards: List[float],
    global_mean: float,
    global_std: float,
) -> List[float]:
    """
    Global GRPO advantage: normalize against a running global mean/std.
    """
    import numpy as np
    if not rewards:
        return []
    r = np.array(rewards, dtype=np.float64)
    normalized = (r - global_mean) / (global_std + 1e-8)
    return [float(np.tanh(x)) for x in normalized]
