"""
Web search for HyperSkill websearch tasks, served by a Gemini knowledge model
via OpenRouter (the same client and key as the backbone LLM).
"""

from __future__ import annotations

from dataclasses import replace

KNOWLEDGE_MODEL = "google/gemini-3.8-flash"
NO_RESULTS = "(No search results found)"
KNOWLEDGE_SYSTEM = (
    "You are a knowledge source standing in for a web search engine. "
    "Given a search query, return up to {n} concise, factual snippets relevant to it. "
    "Format each snippet as '**<short title>**' on one line followed by 1-3 sentences of fact. "
    "Separate snippets with a blank line. Do not speculate; if you have no relevant "
    "knowledge, reply exactly '(No search results found)'."
)


async def knowledge_search(query: str, num_results: int = 5) -> str:
    """Return search-style knowledge snippets for `query` from the Gemini knowledge model."""
    from .config import LLMConfig
    from .llm import async_llm_call

    cfg = replace(LLMConfig(), model=KNOWLEDGE_MODEL)
    text = await async_llm_call(
        f"Search query: {query}", cfg,
        system=KNOWLEDGE_SYSTEM.format(n=num_results),
        use_cache=False, max_tokens=512, temperature=0.0,
    )
    return text.strip() or NO_RESULTS
