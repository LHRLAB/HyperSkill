"""
Self-consistency at evaluation (`hypergraph.vote_k`, off by default).

The greedy rollout plus `vote_k` rollouts sampled at `vote_temperature`
(default 0.7). The final answer is the strict majority of the k sampled
answers (option letter / integer / normalised short span); a tie or no
extractable answer falls back to the greedy rollout. The rollout that carries
the chosen answer is returned (and scored).

With `vote_skip_refusals` a sampled answer that declines to answer ("unknown",
"cannot be determined", ...) does not vote (not applied to option letters).
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..grpo.rewards import normalize_answer

# Seed salts of the sampled rollouts. A salt enters the sampling seed of every
# request of its rollout (llm._call_seed), so the strings are kept exactly as in
# the runs that produced the reported numbers.
SEED_SALT = {False: "hs27vote", True: "hs28vote"}       # keyed by vote_skip_refusals

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def last_answer(text: str) -> Optional[str]:
    m = _ANSWER.findall(text or "")
    return m[-1].strip() if m else None


def _boxed_spans(text: str) -> List[Tuple[int, str]]:
    """(start, content) of every \\boxed{...} (nested braces matched)."""
    out = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            out.append((m.start(), text[m.end():i - 1]))
    return out


def norm_math(s: Optional[str]) -> Optional[str]:
    """Normalised final math answer: integers without sign noise / commas / $ / text."""
    if s is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\boxed\s*\{(.*)\}", r"\1", s)
    s = s.replace("$", "").replace("\\,", "").replace(",", "").replace("\\!", "")
    s = re.sub(r"\^\s*\{?\\circ\}?", "", s)
    s = s.strip().rstrip(".").strip()
    if re.fullmatch(r"[+-]?\d+", s):
        return str(int(s))
    return s or None


def strict_math_answer(text: str) -> Optional[str]:
    """The LAST committed final answer of `text`: whichever of <answer>...</answer> and
    \\boxed{...} occurs last; None when there is neither."""
    text = text or ""
    cands = [(m.start(), m.group(1)) for m in _ANSWER.finditer(text)] + _boxed_spans(text)
    if not cands:
        return None
    return norm_math(max(cands, key=lambda c: c[0])[1])


def parse_options(question: str) -> List[Tuple[str, str]]:
    """[(letter, option text)] of a multiple-choice question ("(A) A. text" or "A. text" lines)."""
    opts = re.findall(r"^\s*\(([A-J])\)\s*(?:[A-J][\.\)]\s*)?(.+?)\s*$", question or "", re.M)
    if not opts:
        opts = re.findall(r"^\s*([A-J])[\.\)]\s+(.+?)\s*$", question or "", re.M)
    return opts


def letter_of(content: str, options: List[Tuple[str, str]]) -> Optional[str]:
    """Option letter named by an answer string, or None."""
    c = (content or "").strip().strip("*").strip()
    if not c:
        return None
    m = re.fullmatch(r"\(?([A-J])\)?[\.\):]?", c)
    if m:
        return m.group(1)
    n = normalize_answer(c)
    exact = [L for L, t in options if normalize_answer(t) == n]
    if len(set(exact)) == 1:
        return exact[0]
    m = re.match(r"^(?:option\s+)?\(?([A-J])\)?[\.\):]\s*\S", c, re.I) or re.match(r"^\(([A-J])\)\s*\S", c)
    if m and m.group(1).isupper():
        return m.group(1)
    m = re.search(r"(?i:answer|option|choice)\s*(?:is)?\s*[:=]?\s*\**\(?([A-J])\b", c)
    if m:
        return m.group(1)
    inside = [L for L, t in options if normalize_answer(t) and normalize_answer(t) in n]
    if len(set(inside)) == 1:
        return inside[0]
    return None


def mc_letter(text: str, options: List[Tuple[str, str]]) -> Optional[str]:
    """Option letter of a response: from its last <answer>; untagged -> the last
    "answer is L" / "Answer: L" phrase of its tail; else None."""
    a = last_answer(text)
    if a is not None:
        return letter_of(a, options)
    ms = list(re.finditer(r"(?i:answer|option|choice)\s*(?:is)?\s*[:=]?\s*\**\(?([A-J])\b", (text or "")[-400:]))
    return ms[-1].group(1) if ms else None


def answer_kind(task_type: str, is_mc: bool) -> str:
    return "math" if task_type == "math" else ("mc" if is_mc else "span")


# ---------------------------------------------------------------------------
# Refusal detection (vote_skip_refusals)
# ---------------------------------------------------------------------------

_REFUSAL_EXACT = {"unknown", "none", "na", "n a", "not sure", "unsure", "unclear", "uncertain",
                  "not known", "cannot determine", "cannot be determined", "undetermined",
                  "not available", "not applicable", "no answer", "not answerable", "unanswerable",
                  "insufficient information", "no information", "not enough information"}
_REFUSAL_PHRASES = [r"\bunable to\b", r"\bcannot be (?:determined|identified|answered|confirmed|found)",
                    r"\bcan ?not (?:determine|identify|answer|tell|confirm|find)",
                    r"\bcan't (?:be )?(?:determine|determined|identify|identified|answer|tell)",
                    r"\b(?:not|im) possible to (?:determine|identify|tell|answer)",
                    r"\bimpossible to (?:determine|identify|tell|answer)",
                    r"\bno (?:specific |relevant |sufficient )?information\b",
                    r"\b(?:not enough|insufficient) (?:information|evidence|details|context)",
                    r"\bdoes not (?:contain|provide|specify|show|mention)",
                    r"\bdo not (?:know|have (?:enough|access|information))", r"\bdon't know\b",
                    r"\bnot (?:specified|provided|mentioned|identifiable|visible|clear from)",
                    r"\bis unknown\b", r"\bremains? unknown\b", r"\bnot (?:be )?answerable\b"]


def is_refusal(answer: Optional[str]) -> bool:
    """True when an answer string declines to answer (<= 40 words)."""
    if answer is None:
        return False
    n = normalize_answer(answer)
    if not n:
        return True
    if n in _REFUSAL_EXACT:
        return True
    if len(n.split()) > 40:
        return False
    low = answer.lower().replace("’", "'")
    return any(re.search(p, low) for p in _REFUSAL_PHRASES)


def refusal_response(text: str) -> bool:
    """The final <answer> of a response is a refusal (no <answer> -> False)."""
    return is_refusal(last_answer(text))


# ---------------------------------------------------------------------------
# Vote
# ---------------------------------------------------------------------------

def vote_key(text: str, kind: str, options: List[Tuple[str, str]], skip_refusals: bool = False) -> Optional[str]:
    if skip_refusals and kind != "mc" and refusal_response(text):
        return None                                   # refusals do not vote
    if kind == "math":
        return strict_math_answer(text)
    if kind == "mc":
        return mc_letter(text, options)
    a = last_answer(text)
    if a is None:
        return None
    return normalize_answer(a) or None


def majority(keys: List[Optional[str]]) -> Tuple[Optional[str], bool]:
    """(winner, tie): the most frequent non-None key; tie=True when several keys share
    the top count; (None, False) when no key."""
    cnt = Counter(k for k in keys if k is not None)
    if not cnt:
        return None, False
    top = max(cnt.values())
    best = [k for k, v in cnt.items() if v == top]
    return (best[0], False) if len(best) == 1 else (None, True)


def _slim(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: r.get(k) for k in ("full_response", "response_list", "end", "fail", "finish", "final_turn")}


async def vote_rollouts(run_one: Callable[[Optional[int]], Awaitable[Dict[str, Any]]], k: int,
                        kind: str, question: str, skip_refusals: bool = False) -> Dict[str, Any]:
    """run_one(None) = greedy rollout, run_one(j) = j-th sampled rollout. Returns the
    rollout carrying the voted answer, with a "vote" record (all k samples)."""
    options = parse_options(question) if kind == "mc" else []
    res = await asyncio.gather(*[run_one(j) for j in [None] + list(range(k))], return_exceptions=True)
    greedy, samples = res[0], res[1:]
    keys = [None if isinstance(s, BaseException) else vote_key(s["full_response"], kind, options, skip_refusals)
            for s in samples]
    g_key = (None if isinstance(greedy, BaseException)
             else vote_key(greedy["full_response"], kind, options, skip_refusals))
    winner, tie = majority(keys)
    if winner is None or winner == g_key:               # tie / no sampled answer -> greedy
        chosen = "greedy"
        if isinstance(greedy, BaseException):
            ok = [j for j, s in enumerate(samples) if not isinstance(s, BaseException)]
            if not ok:
                raise greedy
            chosen = next((j for j in ok if keys[j] is not None), ok[0])
    else:
        chosen = keys.index(winner)
    out = dict(greedy if chosen == "greedy" else samples[chosen])
    out["vote"] = {
        "kind": kind, "k": k, "skip_refusals": skip_refusals, "keys": keys, "greedy_key": g_key,
        "winner": winner, "tie": tie, "chosen": chosen,
        "greedy_error": f"{type(greedy).__name__}: {greedy}"[:200] if isinstance(greedy, BaseException) else None,
        "samples": [{"error": f"{type(s).__name__}: {s}"[:200]} if isinstance(s, BaseException) else _slim(s)
                    for s in samples],
    }
    return out
