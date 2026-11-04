"""
Protocol skill (`hypergraph.protocol_skill`).

A weaker backbone often fails the agent harness's output protocol rather than
the task itself: it writes a malformed tool call (`<TOOL_TAG>text_search>q`),
keeps calling a tool after the tool budget is spent, emits a tool call although
no tool is offered, or never writes the final `<answer>` tag (e.g. a long
derivation ending in `\\boxed{}`). The harness then has no answer to extract.

HyperSkill learns which of these failure types its policy actually makes from
its own TRAINING rollouts (every rollout is classified when it ends; counts are
stored with the library in `protocol_skill.json`). A failure type becomes part
of the protocol skill once it has been observed in `protocol_min_count` training
rollouts. With `protocol_skill: true` the skill is

  * injected together with the retrieved skills for every question; it is not
    gated by retrieval similarity, since it describes the task's protocol and
    not the question;
  * re-injected after each tool turn with the remaining tool budget, through the
    "--- Updated Skills ---" block that training already uses to refresh the
    retrieved skills mid-rollout.

The shared prompts and the answer parser are unchanged; the skill only adds
instructions to the injected context. Each failure type maps to one fixed
instruction line (below), filled in with the dataset's tool tags and budget.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

FAILURE_TYPES = ("malformed_tool", "over_budget", "tool_without_tools", "no_answer")
PROTOCOL_FILE = "protocol_skill.json"

_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_TOOLISH_RE = re.compile(r"TOOL_TAG|<\s*/?\s*\w*search\w*\s*>|\w+_search\s*>", re.IGNORECASE)


def classify_rollout(response_list: List[str], tool_tags: Iterable[str],
                     end_reason: str) -> Optional[str]:
    """Failure type of a finished rollout, or None if it ended with an answer.

    end_reason (set by the rollout loop): "answer" | "budget" (tool budget
    spent, the last response asked for more) | "no_action" (no valid tool call
    and no answer) | "repeat" (repeated query) | "length" (prompt too long) |
    "rounds" (round limit)."""
    final = response_list[-1] if response_list else ""
    if _ANSWER_RE.search(final or ""):
        return None
    tool_tags = list(tool_tags)
    if not tool_tags:
        return "tool_without_tools" if _TOOLISH_RE.search(final or "") else "no_answer"
    if end_reason in ("budget", "repeat"):
        return "over_budget"
    if end_reason == "no_action" and _TOOLISH_RE.search(final or ""):
        return "malformed_tool"
    return "no_answer"


_SHAPE_RE = re.compile(r"<\s*TOOL_TAG\s*>(?:\s*<\s*/?\s*[\w\-]+\s*>|\s*[\w\-]+\s*>)?"
                       r"|<\s*/?\s*[\w\-]*search[\w\-]*\s*>?|\b[\w\-]+_search\s*>", re.IGNORECASE)


def malformed_shape(response_list: List[str]) -> Optional[str]:
    """Opening of the malformed tool call in the final response, whitespace removed
    (e.g. "<TOOL_TAG>text_search>"); the query itself is not kept. A token after
    <TOOL_TAG> is kept only if it is tag-like (ends with ">"), so a query written
    directly inside the placeholder gives "<TOOL_TAG>". Logged in the library only."""
    final = response_list[-1] if response_list else ""
    m = _SHAPE_RE.search(final or "")
    return re.sub(r"\s+", "", m.group(0))[:40] if m else None


class ProtocolStats:
    """Failure-type counts over the training rollouts of one library."""

    def __init__(self, min_count: int = 2):
        self.min_count = int(min_count)
        self.counts: Dict[str, int] = {t: 0 for t in FAILURE_TYPES}
        self.n_rollouts = 0
        self.shapes: Dict[str, int] = {}     # malformed tool-call forms (logged only)

    def add(self, ftype: Optional[str], shape: Optional[str] = None) -> None:
        self.n_rollouts += 1
        if ftype:
            self.counts[ftype] = self.counts.get(ftype, 0) + 1
        if ftype == "malformed_tool" and shape:
            self.shapes[shape] = self.shapes.get(shape, 0) + 1

    @property
    def active(self) -> List[str]:
        return [t for t in FAILURE_TYPES if self.counts.get(t, 0) >= self.min_count]

    def save(self, working_dir: str) -> None:
        p = Path(working_dir) / PROTOCOL_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"n_rollouts": self.n_rollouts, "counts": self.counts,
                                 "min_count": self.min_count, "active": self.active,
                                 "malformed_shapes": self.shapes}, indent=2))

    @classmethod
    def load(cls, working_dir: str, min_count: int = 2) -> Optional["ProtocolStats"]:
        p = Path(working_dir) / PROTOCOL_FILE
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        st = cls(min_count)
        st.n_rollouts = int(d.get("n_rollouts", 0))
        st.counts.update({k: int(v) for k, v in (d.get("counts") or {}).items()})
        st.shapes = {k: int(v) for k, v in (d.get("malformed_shapes") or {}).items()}
        return st


def _examples(tool_tags: List[str]) -> str:
    return " or ".join(f"<{t}>your query</{t}>" for t in tool_tags)


def answer_form(golds: List[str]) -> Dict:
    """Answer form of the training gold answers (protocol_answer_form: data): share
    of option-letter golds, median length in words, and the most frequent
    abstention gold ("Not answerable", "Not mentioned.", ...) if it covers >= 10%
    of the training questions."""
    from collections import Counter
    import statistics
    golds = [str(g).strip() for g in golds if str(g).strip()]
    if not golds:
        return {}
    mc = sum(bool(re.match(r"^\(?[A-J]\)?([.)\s]|$)", g)) for g in golds) / len(golds)
    abst = Counter(g for g in golds if len(g.split()) <= 4 and _ABSTAIN_RE.search(g))
    top, n = abst.most_common(1)[0] if abst else (None, 0)
    return {"mc": mc, "median_words": float(statistics.median(len(g.split()) for g in golds)),
            "frequent": top if mc < 0.5 and n / len(golds) >= 0.10 else None}


_ABSTAIN_RE = re.compile(r"\b(not answerable|unanswerable|not mentioned|not (?:given|provided|stated|available)|"
                         r"cannot be (?:determined|answered)|no answer)\b", re.IGNORECASE)

_SHORT_LINE = ("Always end with your final answer inside <answer>...</answer>, as short as the question "
               "allows (a name, number, short phrase or option letter, not a full sentence).")


def _answer_line(form: Optional[Dict]) -> str:
    if not form:
        return _SHORT_LINE
    if form["mc"] >= 0.5:
        line = ("Always end with only the option letter of your choice inside <answer>...</answer> "
                "(e.g. <answer>B</answer>).")
    elif form["median_words"] <= 3:
        line = _SHORT_LINE
    else:
        line = (f"Always end with your final answer inside <answer>...</answer>, written as one complete sentence "
                f"(about {round(form['median_words'])} words) that restates the key terms of the question "
                f"together with the answer.")
    if form.get("frequent"):
        line += f" If the input does not contain the answer, answer exactly <answer>{form['frequent']}</answer>."
    return line


def protocol_text(active: List[str], task_type: str, tool_tags: List[str],
                  budget: int, max_new_tokens: int, form: Optional[Dict] = None) -> str:
    """The protocol skill injected before the question ("" if nothing is active).
    `form` (protocol_answer_form: data) replaces the fixed short-answer line by
    one derived from the training gold answers (answer_form)."""
    if not active:
        return ""
    lines = []
    if tool_tags and "malformed_tool" in active:
        lines.append(f"To call a tool, put the query inside that tool's own tag, exactly like {_examples(tool_tags)}; "
                     f"the tag name is the tool name (never write TOOL_TAG and never nest tags).")
    if tool_tags and ("over_budget" in active or "malformed_tool" in active):
        lines.append(f"Use at most {budget} tool calls in total; after the last one, or as soon as you can answer, "
                     f"stop calling tools and give your best final answer.")
    if not tool_tags and "tool_without_tools" in active:
        lines.append("No tool is available for this question: do not write any tool call, search request or "
                     "TOOL_TAG; answer directly from the given input.")
    if task_type == "math":
        words = max(200, int(max_new_tokens * 0.45))
        lines.append(f"Reason concisely (the whole response must stay well under {words} words) and end with the "
                     f"final integer inside <answer>...</answer>; do not use \\boxed{{}}.")
    else:
        lines.append(_answer_line(form))
    return "Output protocol (learned from training failures): " + " ".join(lines)


def reinject_text(tool_tags: List[str], used: int, budget: int) -> str:
    """Protocol reminder appended after a tool turn."""
    left = budget - used
    if left > 0:
        return (f"Protocol: you may make {left} more tool call(s), written as {_examples(tool_tags)}; "
                f"if the results above suffice, give the final answer inside <answer>...</answer> now.")
    return (f"Protocol: you have used all {budget} tool calls. Do not call a tool again; give your best "
            f"final answer now inside <answer>...</answer>.")
