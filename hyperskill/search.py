"""
Web search via SerpAPI for HyperSkill websearch tasks.

API key is read from api_key_serpapi.txt in the project root,
or from the SERPAPI_KEY environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_serpapi_key() -> str:
    key_file = Path(__file__).parent.parent / "api_key_serpapi.txt"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return os.environ.get("SERPAPI_KEY", "")


async def serpapi_search(
    query: str,
    api_key: str,
    num_results: int = 5,
) -> str:
    """
    Search via SerpAPI and return a formatted results string.

    Returns a multi-line string with title, snippet, and URL for each result.
    Raises httpx.HTTPError on network/API failure (caller should catch).
    """
    import httpx

    params = {
        "q": query,
        "api_key": api_key,
        "num": num_results,
        "engine": "google",
    }
    async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
        resp = await client.get("https://serpapi.com/search.json", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("organic_results", [])
    if not results:
        return "(No search results found)"

    lines = []
    for r in results[:num_results]:
        title   = r.get("title", "")
        snippet = r.get("snippet", "")
        link    = r.get("link", "")
        lines.append(f"**{title}**\n{snippet}\n{link}")
    return "\n\n".join(lines)


async def serpapi_image_search(
    query: str,
    api_key: str,
    num_results: int = 5,
) -> str:
    """Image search via SerpAPI (Google Images text-based image search)."""
    import httpx

    params = {
        "engine": "google_images",
        "q": query,
        "api_key": api_key,
        "num": num_results,
    }

    async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
        resp = await client.get("https://serpapi.com/search.json", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("images_results", [])
    if not results:
        return "(No image results found)"
    lines = []
    for r in results[:num_results]:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        source = r.get("source", "")
        lines.append(f"**{title}**\n{snippet}\n{source}")
    return "\n\n".join(lines)
