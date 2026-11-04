"""
Prompt and tool-call options of the HyperSkill agent (all off by default).

  concrete_tool_prompt    The search format names the dataset's real tool tag and
                          shows one concrete call instead of the TOOL_TAG
                          placeholder; without tools (Omni) the prompt has no
                          search format at all.
  final_answer_turn       When a rollout would end without a closed <answer> in
                          its last response (tool budget spent, repeated query,
                          no valid action, round / length limit), the agent is
                          asked once more for its final answer. The protocol
                          statistics still classify the rollout as it ended
                          before this turn.
  answer_suffix           Answer form after the question: math -> the integer
                          inside <answer> (no \\boxed{}); multiple choice (>= 50%
                          option-letter training golds) -> the option letter
                          inside <answer>.
  tool_parse_lenient      The tool-call parser also executes the placeholder
                          forms the shared prompt induces:
                          <TOOL_TAG>text_search>q</text_search>,
                          <TOOL_TAG>text_search>q</TOOL_TAG>, and, when the
                          dataset has exactly one tool, <TOOL_TAG>q</TOOL_TAG>.
  doc_own_only            doc_search for a question whose own PDF has no indexed
                          text returns an explicit message instead of a search.

Token / word budgets (rollout.max_new_tokens, rollout.max_prompt_length) are
configuration only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# concrete_tool_prompt
# ---------------------------------------------------------------------------

SEARCH_AGENT_PROMPT_V2 = """\
You are an expert research assistant equipped with search tools.
Analyze the question and decide whether to search for more information or to
provide a final answer directly.
{tool_desc}

You must respond in ONE of the following formats:

  Format A — Direct answer (when confident):
    <reason>...</reason>
    <answer>...</answer>

  Format B — Search for more information:
    <reason>...</reason>
    {tool_call}

{tool_example}
"""

OMNI_AGENT_PROMPT_V2 = """\
You are an expert multimodal reasoning assistant. You can process text,
images, audio transcripts, and video descriptions. Analyze all provided
modalities together to derive a precise answer.
{tool_desc}

You must respond in ONE of the following formats:

  Format A — Direct answer (when confident):
    <reason>...</reason>
    <answer>...</answer>

  Format B — Search for more information:
    <reason>...</reason>
    {tool_call}

{tool_example}
"""

OMNI_AGENT_PROMPT_V2_NO_TOOLS = """\
You are an expert multimodal reasoning assistant. You can process text,
images, audio transcripts, and video descriptions. Analyze all provided
modalities together to derive a precise answer.

You must respond in the following format:
    <reason>...</reason>
    <answer>...</answer>
"""

_TOOL_EXAMPLE_QUERY = {
    "text_search": "Eiffel Tower height in metres",
    "doc_search": "total revenue in fiscal year 2019",
}


def _tool_tags(tool_desc: str) -> List[str]:
    """Tool tags listed in a format_available_tools() description, in order."""
    return list(dict.fromkeys(re.findall(r"<(\w+)>your query</\1>", tool_desc or "")))


def _tool_format(tool_desc: str) -> Dict[str, str]:
    tags = _tool_tags(tool_desc)
    call = " or ".join(f"<{t}>specific query</{t}>" for t in tags)
    ex = [f"<{t}>{_TOOL_EXAMPLE_QUERY.get(t, 'specific query')}</{t}>" for t in tags]
    example = "The tag is the tool's own name. Example of a correct tool call: " + " ; ".join(ex)
    return {"tool_call": call, "tool_example": example}


def initial_prompt_v2(question: str, modality: str, tool_desc: str = "") -> str:
    from ..prompts import REASONING_PROMPT
    if modality == "omni":
        base = (OMNI_AGENT_PROMPT_V2.format(tool_desc=tool_desc, **_tool_format(tool_desc))
                if tool_desc else OMNI_AGENT_PROMPT_V2_NO_TOOLS)
    elif tool_desc:
        base = SEARCH_AGENT_PROMPT_V2.format(tool_desc=tool_desc, **_tool_format(tool_desc))
    else:
        base = REASONING_PROMPT
    return base + f"\n\nHere is the question:\n{question}"


# ---------------------------------------------------------------------------
# answer_suffix
# ---------------------------------------------------------------------------

MATH_SUFFIX = ("Reason step by step, then end your response with the final integer inside "
               "<answer>...</answer> (for example <answer>N</answer> for the integer N); do not use \\boxed{}.")
MC_SUFFIX = ("End your response with only the letter of the correct option inside <answer>...</answer> "
             "(for example <answer>X</answer> for option X).")


def answer_suffix(task_type: str, is_mc: bool) -> str:
    if task_type == "math":
        return MATH_SUFFIX
    if is_mc:
        return MC_SUFFIX
    return ""


def join_suffix(a: str, b: str) -> str:
    return f"{a} {b}" if a and b else (a or b)


def build_prompt(question: str, modality: str, skills: list, tool_desc: str = "",
                 skills_context: Optional[str] = None, v2: bool = False, suffix: str = "") -> str:
    """HyperSkill's initial prompt. With v2 off and no suffix this is
    prompts.build_skill_augmented_prompt byte for byte."""
    from ..prompts import (build_skill_augmented_prompt, build_initial_prompt,
                           SKILL_AUGMENTED_PROMPT, format_skills_context)
    if not v2 and not suffix:
        return build_skill_augmented_prompt(question=question, modality=modality, skills=skills,
                                            tool_desc=tool_desc, skills_context=skills_context)
    base = (initial_prompt_v2 if v2 else build_initial_prompt)(question, modality, tool_desc=tool_desc)
    if suffix:
        base = base + "\n\n" + suffix
    if skills_context:
        return SKILL_AUGMENTED_PROMPT.format(skills_context=skills_context) + "\n\n" + base
    if not skills:
        return base
    return SKILL_AUGMENTED_PROMPT.format(skills_context=format_skills_context(skills)) + "\n\n" + base


# ---------------------------------------------------------------------------
# final_answer_turn
# ---------------------------------------------------------------------------

_CLOSED_ANSWER = re.compile(r"<answer>.*?</answer>", re.DOTALL)


def has_closed_answer(text: str) -> bool:
    return bool(_CLOSED_ANSWER.search(text or ""))


def final_turn_text(end_reason: str, task_type: str, is_mc: bool) -> str:
    if task_type == "math":
        form = " (only the final integer)"
    elif is_mc:
        form = " (only the option letter)"
    else:
        form = ""
    if end_reason in ("budget", "repeat"):
        head = ("Tool budget exhausted: no further tool call will be executed. "
                "Based on the information above, give your final answer now")
    else:
        head = "You have not given your final answer yet. Give it now"
    return f"\n\n{head} inside <answer>...</answer>{form}.\n"


# ---------------------------------------------------------------------------
# tool_parse_lenient
# ---------------------------------------------------------------------------

def parse_tool_call(response: str, tags: List[str], lenient: bool = False) -> Optional[Tuple[str, str]]:
    """(tag, query) of the tool call in `response`, or None.
    Strict form first: the first tool (in registry order) whose <tag>...</tag>
    occurs. `lenient` adds the TOOL_TAG placeholder forms."""
    for tag in tags:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
        if m:
            return tag, m.group(1).strip()
    if not lenient or not tags:
        return None
    alt = "|".join(re.escape(t) for t in tags)
    m = re.search(rf"<\s*TOOL_TAG\s*>\s*<?\s*({alt})\s*>(.*?)(?:<\s*/\s*(?:TOOL_TAG|{alt})\s*>|\Z)",
                  response, re.DOTALL)
    if m:
        q = re.sub(r"<\s*/?\s*(?:TOOL_TAG|" + alt + r")\s*>", " ", m.group(2)).strip()
        return (m.group(1), q) if q else None
    if len(tags) == 1:
        m = re.search(r"<\s*TOOL_TAG\s*>(.*?)<\s*/\s*TOOL_TAG\s*>", response, re.DOTALL)
        if m and m.group(1).strip():
            return tags[0], m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# doc_own_only
# ---------------------------------------------------------------------------

def doc_no_text_message(tool, resource_paths: List[str]) -> Optional[str]:
    """None if the question's own PDF(s) have indexed text; otherwise the message
    returned to the agent instead of a search."""
    tool._ensure_index()
    stems = {Path(p).stem for p in resource_paths}
    seen = tool.__dict__.setdefault("_has_text_memo", {})      # per-stem memo on the tool instance
    for s in stems - set(seen):
        seen[s] = any(cid.startswith(f"doc_{s}_") for cid in tool._chunk_ids)
    if any(seen[s] for s in stems):
        return None
    names = ", ".join(sorted(Path(p).name for p in resource_paths))
    return (f"(No text could be extracted from the document {names}: it has no text layer, "
            f"so document search returns nothing for it.)")
