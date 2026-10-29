"""
Reward functions for Training-Free GRPO in HyperSkill.

Two reward signals:
  1. F1 reward       — token-level F1 between predicted and ground-truth answers
  2. Format reward   — structural validity of the response format (logged only)
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


_MC_LETTER_RE = re.compile(r"\b([A-J])\b")

def _extract_mc_letter(s: str) -> Optional[str]:
    """Extract a multiple-choice letter (A-J) from any text.

    Handles all common surface forms the model emits: `A`, `(A)`, `A.`,
    `A. some text`, `Answer: A`, `the answer is A`, etc. Returns the bare
    letter, or None if the string is clearly not a single-letter MC answer.
    """
    if not s:
        return None
    s = s.strip()
    # Fast paths for the most common forms
    m = re.match(r"^\(?([A-J])\)?[\.\)\s]?.*$", s)
    if m:
        return m.group(1)
    # "the answer is X" / "Answer: X"
    m = re.search(r"(?:answer(?:\s+is)?|option|choice)\s*[:=]?\s*\(?([A-J])\b",
                  s, re.IGNORECASE)
    if m:
        return m.group(1)
    # Last-resort: first standalone letter token
    m = _MC_LETTER_RE.search(s)
    return m.group(1) if m else None


def _gold_is_mc(golds: List[str]) -> Optional[str]:
    """If gold answers look MC (e.g. "A", "A.", "A. Four pieces"), return the letter."""
    for g in golds:
        g = g.strip()
        m = re.match(r"^\(?([A-J])\)?[\.\)\s]", g) or re.match(r"^\(?([A-J])\)?$", g)
        if m:
            return m.group(1)
    return None


def _entity_containment(pred: str, gold: str) -> float:
    """Score a short-entity gold answer by containment in a longer prediction.

    Open-ended QA datasets (FVQA, InfoSeek, LiveVQA…) evaluate answers whose
    gold is a 1-5 word named entity. A correct model response often embeds the
    entity in an explanatory sentence, which token F1 dilutes heavily. If the
    normalized gold appears verbatim as a substring / full token match in the
    normalized prediction we award 1.0; otherwise fall back to token F1.
    """
    g = normalize_answer(gold)
    p = normalize_answer(pred)
    if not g:
        return 0.0
    gold_tokens = g.split()
    pred_tokens = p.split()
    if len(gold_tokens) <= 5:
        # full substring match
        if g in p:
            return 1.0
        # all gold tokens present in pred tokens (ignoring order)
        if set(gold_tokens).issubset(set(pred_tokens)):
            return 1.0
    return token_f1(pred, gold)


def compute_f1_reward(
    prediction: str,
    gold_answers: Union[str, List[str]],
) -> float:
    """
    Answer-shape-aware F1 reward:
      1. If gold is MC (letter) → strict letter match (0.0 / 1.0).
      2. If gold is a short entity (≤5 tokens) → substring/token-set containment
         earns 1.0; otherwise falls back to token F1.
      3. Else → token-level F1.

    Returns max score over all gold alternatives.
    """
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]

    # Extract predicted answer from <answer>...</answer> tag
    answer_match = re.findall(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        pred_text = answer_match[-1].strip()
    else:
        pred_text = prediction.strip()

    # -- 1. MC shape -----------------------------------------------------
    gold_letter = _gold_is_mc(gold_answers)
    if gold_letter is not None:
        pred_letter = _extract_mc_letter(pred_text)
        if pred_letter and pred_letter == gold_letter:
            return 1.0
        # Some golds list the letter+text; still accept full-text F1 as fallback
        return max(token_f1(pred_text, g) for g in gold_answers)

    # -- 2/3. Entity containment / token F1 ------------------------------
    return max(_entity_containment(pred_text, g) for g in gold_answers)


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
_DOC_SEARCH_RE = re.compile(
    r"^<reason>.*?</reason>.*?<doc_search>.*?</doc_search>$", re.DOTALL
)
_MEDIA_SEARCH_RE = re.compile(
    r"^<reason>.*?</reason>.*?<media_search>.*?</media_search>$", re.DOTALL
)


def _is_valid_doc_search(response: str) -> bool:
    if not _DOC_SEARCH_RE.match(response):
        return False
    if response.count("<reason>") != 1 or response.count("</reason>") != 1:
        return False
    if response.count("<doc_search>") != 1 or response.count("</doc_search>") != 1:
        return False
    if "<answer>" in response or "</answer>" in response:
        return False
    return True


def _is_valid_media_search(response: str) -> bool:
    if not _MEDIA_SEARCH_RE.match(response):
        return False
    if response.count("<reason>") != 1 or response.count("</reason>") != 1:
        return False
    if response.count("<media_search>") != 1 or response.count("</media_search>") != 1:
        return False
    if "<answer>" in response or "</answer>" in response:
        return False
    return True


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
            if not (_is_valid_text_search(resp)
                    or _is_valid_image_search(resp)
                    or _is_valid_doc_search(resp)
                    or _is_valid_media_search(resp)
                    or _is_valid_direct_answer(resp)):
                if "<reason>" in resp:
                    return 0.5
                return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def compute_combined_reward(
    prediction: str,
    gold_answers: Union[str, List[str]],
) -> float:
    """Task reward used for training and evaluation (= compute_f1_reward)."""
    # Unified: all tasks use pure token-level F1 on the extracted answer
    return compute_f1_reward(prediction, gold_answers)

