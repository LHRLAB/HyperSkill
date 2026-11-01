"""
Unified tool registry for HyperSkill agents.

Each dataset category maps to a set of available tools:
  Text-Math  (AIME2024, AIME2025)       → no tools (pure reasoning)
  Text-Web   (WebWalkerQA)               → text_search (Gemini knowledge model)
  VL-Web     (FVQA, InfoSeek, ...)       → text_search (Gemini knowledge model)
  VL-Doc     (DocBench, MMLongBench-Doc) → doc_search (local corpus VDB)
  Omni       (OmniVideoBench, WorldSense)→ no tools (frames/audio sent to the LLM)

All tools share a unified interface:
  async def search(query: str, ...) -> str
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Tool base
# ═══════════════════════════════════════════════════════════════════════════

class SearchTool:
    """Base class for all search tools (registered under their XML tag)."""
    name: str = "base"

    async def search(self, query: str) -> str:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge-search on-disk cache
#
# All knowledge-model search calls are cached under
# cache/gemini_knowledge/text/<sha256(key)>.txt, where the key covers the
# knowledge model, its system prompt, num_results and the lowercased query.
# Empty answers (NO_RESULTS) and errors are not cached.
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "cache" / "gemini_knowledge"


def _cache_path(kind: str, key: str) -> Path:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    d = _CACHE_ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.txt"


def _cache_get(kind: str, key: str) -> Optional[str]:
    p = _cache_path(kind, key)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def _cache_put(kind: str, key: str, value: str) -> None:
    try:
        # Write-then-rename: parallel dataset subprocesses share this cache.
        p = _cache_path(kind, key)
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        tmp.write_text(value, encoding="utf-8")
        os.replace(tmp, p)
    except OSError as exc:
        logger.warning("search cache write failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Text search (Gemini knowledge model)
# ═══════════════════════════════════════════════════════════════════════════

class TextSearchTool(SearchTool):
    """Text search served by the Gemini knowledge model (with disk cache)."""
    name = "text_search"

    num_results = 5

    async def search(self, query: str) -> str:
        from ..search import KNOWLEDGE_MODEL, KNOWLEDGE_SYSTEM, NO_RESULTS, knowledge_search
        cache_key = json.dumps(
            {"model": KNOWLEDGE_MODEL, "system": KNOWLEDGE_SYSTEM,
             "num_results": self.num_results, "query": query.strip().lower()},
            ensure_ascii=False, sort_keys=True,
        )
        cached = _cache_get("text", cache_key)
        if cached is not None:
            return cached

        try:
            result = await knowledge_search(query, num_results=self.num_results)
        except Exception as exc:
            logger.warning("text_search failed: %s", exc)
            return f"(Search error: {exc})"
        # An empty completion (e.g. max_tokens spent on reasoning, filtered
        # output) is indistinguishable from "no knowledge" here; don't pin it.
        if result != NO_RESULTS:
            _cache_put("text", cache_key, result)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Document corpus search (local VDB)
# ═══════════════════════════════════════════════════════════════════════════

class DocSearchTool(SearchTool):
    """BM25 search over a local document corpus (PDF pages → chunks)."""
    name = "doc_search"

    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self._chunks: List[str] = []
        self._chunk_ids: List[str] = []

    def _ensure_index(self):
        """Build BM25 index from PDF/text files (no embedding, instant)."""
        if self._chunks:
            return

        chunks_path = self.corpus_dir / ".doc_search_chunks.json"

        # Load cache
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._chunks = data["chunks"]
                self._chunk_ids = data["ids"]
            return

        resources = self.corpus_dir / "resources"
        if not resources.exists():
            return

        print(f"  [DocSearch] Building BM25 index for {self.corpus_dir.name}...")
        chunks = []
        chunk_ids = []

        for f in sorted(resources.iterdir()):
            if f.suffix.lower() == ".pdf":
                page_texts = _extract_pdf_text(f)
                for i, text in enumerate(page_texts):
                    if text.strip():
                        cid = f"doc_{f.stem}_p{i}"
                        for j, chunk in enumerate(_chunk_text(text, 512)):
                            chunks.append(chunk)
                            chunk_ids.append(f"{cid}_c{j}")
            elif f.suffix.lower() in (".txt", ".md", ".json"):
                text = f.read_text(encoding="utf-8", errors="ignore")
                for j, chunk in enumerate(_chunk_text(text, 512)):
                    chunks.append(chunk)
                    chunk_ids.append(f"doc_{f.stem}_c{j}")

        self._chunks = chunks
        self._chunk_ids = chunk_ids

        if chunks:
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunks, "ids": chunk_ids}, f, ensure_ascii=False)
            print(f"  [DocSearch] Indexed {len(chunks)} chunks")

    async def search(self, query: str, top_k: int = 5,
                     restrict_to: Optional[List[str]] = None) -> str:
        """
        Args:
            restrict_to: list of PDF stems (e.g. ["STEPBACK"]) — when given,
                BM25 only ranks chunks whose chunk_id starts with `doc_<stem>_`.
                Used by per-question doc QA so search doesn't drift to
                unrelated PDFs in the same corpus.
        """
        self._ensure_index()
        if not self._chunks:
            return "(No document corpus available)"

        chunks, chunk_ids = self._chunks, self._chunk_ids
        if restrict_to:
            stems = {Path(p).stem for p in restrict_to}
            keep = [i for i, cid in enumerate(chunk_ids)
                    if any(cid.startswith(f"doc_{stem}_") for stem in stems)]
            if keep:
                chunks    = [self._chunks[i]    for i in keep]
                chunk_ids = [self._chunk_ids[i] for i in keep]

        results = _bm25_search(query, chunks, chunk_ids, top_k)
        if not results:
            return "(No matching documents found)"

        lines = []
        for cid, score, text in results:
            lines.append(f"[{cid}] (BM25={score:.1f})\n{text[:500]}")
        return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Tool registry
# ═══════════════════════════════════════════════════════════════════════════

# task_type → list of tool classes to instantiate
TOOL_PROFILES = {
    "math":        [],                                # pure reasoning
    "webagent":    ["text_search"],                    # WebWalkerQA
    "websearch":   ["text_search"],                    # VL-Web datasets
    "mmreasoning": [],                                 # default fallback
    "docsearch":   ["doc_search"],                     # VL-Doc datasets
}

_tool_cache: Dict[str, Dict[str, SearchTool]] = {}


def build_tools(
    task_type: str,
    dataset_name: str = "",
    corpus_dir: str = "",
) -> Dict[str, SearchTool]:
    """
    Build the tool set for a given task type / dataset.
    Instances are cached per (dataset_name, corpus_dir) so expensive
    indexes (DocSearch) are built only once.

    Returns:
        dict mapping tool tag → SearchTool instance
    """
    cache_key = f"{dataset_name}::{corpus_dir}"
    if cache_key in _tool_cache:
        return _tool_cache[cache_key]

    tool_names = TOOL_PROFILES.get(task_type, [])

    tools: Dict[str, SearchTool] = {}
    for name in tool_names:
        if name == "text_search":
            tools["text_search"] = TextSearchTool()
        elif name == "doc_search":
            tools["doc_search"] = DocSearchTool(corpus_dir)

    _tool_cache[cache_key] = tools
    return tools


def format_available_tools(tools: Dict[str, SearchTool]) -> str:
    """Format available tools for the system prompt."""
    if not tools:
        return ""
    descs = {
        "text_search":  "Text search: <text_search>your query</text_search>",
        "doc_search":   "Document search: <doc_search>your query</doc_search>",
    }
    lines = ["\nAvailable tools:"]
    for tag in tools:
        if tag in descs:
            lines.append(f"  - {descs[tag]}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bm25_search(
    query: str,
    chunks: List[str],
    chunk_ids: List[str],
    top_k: int = 5,
) -> List[tuple]:
    """
    Fast BM25 keyword search. No embeddings, no API calls.
    Returns list of (chunk_id, score, text).
    """
    import math
    query_terms = set(query.lower().split())
    if not query_terms:
        return []

    # Term frequencies
    N = len(chunks)
    k1, b = 1.5, 0.75
    avg_dl = sum(len(c.split()) for c in chunks) / max(N, 1)

    # Document frequency per term
    df = {}
    for term in query_terms:
        df[term] = sum(1 for c in chunks if term in c.lower())

    scores = []
    for i, chunk in enumerate(chunks):
        words = chunk.lower().split()
        dl = len(words)
        score = 0.0
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        for term in query_terms:
            tf = word_counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            score += idf * tf_norm
        if score > 0:
            scores.append((chunk_ids[i], score, chunk))

    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def _extract_pdf_text(pdf_path: Path) -> List[str]:
    """Extract text from each page of a PDF."""
    try:
        import pymupdf  # PyMuPDF
        doc = pymupdf.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return pages
    except ImportError:
        logger.warning("PyMuPDF not installed, skipping PDF: %s", pdf_path)
        return []
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", pdf_path, exc)
        return []


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

