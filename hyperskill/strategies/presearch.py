"""
Question pre-search (`hypergraph.question_presearch`, web QA, evaluation only).

The question itself (verbatim, whitespace-normalised) is sent once as a
knowledge-search query before the first model call; the results are put after
the question. Not counted in the rollout's tool budget.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

QP_BLOCK = ("\n\nResults of a knowledge search for the question itself (run automatically before your first "
            "step):\n[{tool} Results]\n{results}\n\nUse these results if they answer the question; otherwise "
            "search further.")


def presearch_query(question: str) -> str:
    return " ".join((question or "").split())


async def presearch(tools, item, prompt: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """(prompt with the pre-search block, record); unchanged when the item has no text_search."""
    if "text_search" not in tools:
        return prompt, None
    q = presearch_query(item.question)
    try:
        res = await tools["text_search"].search(query=q)
    except Exception as exc:
        res = f"(Search error: {exc})"
    return prompt + QP_BLOCK.format(tool="text_search", results=res), {"query": q[:500], "results": res[:3000]}
