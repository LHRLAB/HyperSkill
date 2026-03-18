"""
Unified tool registry for HyperSkill agents.

Each dataset category maps to a set of available tools:
  Text-Math  (AIME2024, AIME2025)       → no tools (pure reasoning)
  Text-Web   (WebWalkerQA)               → text_search (SerpAPI)
  VL-Web     (FVQA, InfoSeek, ...)       → text_search + image_search (SerpAPI)
  VL-Doc     (DocBench, MMLongBench-Doc) → doc_search (local corpus VDB)
  Omni       (OmniVideoBench, WorldSense)→ media_search (local audio/video RAG)

All tools share a unified interface:
  async def search(query: str, **kwargs) -> str
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Tool base
# ═══════════════════════════════════════════════════════════════════════════

class SearchTool:
    """Base class for all search tools."""
    name: str = "base"
    tag: str = "search"  # XML tag in LLM output

    async def search(self, query: str, **kwargs) -> str:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# SerpAPI text search
# ═══════════════════════════════════════════════════════════════════════════

class TextSearchTool(SearchTool):
    """Web text search via SerpAPI."""
    name = "text_search"
    tag = "text_search"

    def __init__(self, api_key: str = ""):
        from ..search import _resolve_serpapi_key
        self.api_key = api_key or _resolve_serpapi_key()

    async def search(self, query: str, **kwargs) -> str:
        if not self.api_key:
            return f"(SerpAPI key not configured — no results for: {query})"
        from ..search import serpapi_search
        try:
            return await serpapi_search(query, self.api_key, num_results=5)
        except Exception as exc:
            logger.warning("text_search failed: %s", exc)
            return f"(Search error: {exc})"


# ═══════════════════════════════════════════════════════════════════════════
# SerpAPI image search (reverse image / image query)
# ═══════════════════════════════════════════════════════════════════════════

class ImageSearchTool(SearchTool):
    """Image-based web search via SerpAPI (Google Lens)."""
    name = "image_search"
    tag = "image_search"

    def __init__(self, api_key: str = ""):
        from ..search import _resolve_serpapi_key
        self.api_key = api_key or _resolve_serpapi_key()

    async def search(self, query: str, image_url: str = "", **kwargs) -> str:
        if not self.api_key:
            return "(SerpAPI key not configured)"
        from ..search import serpapi_image_search
        try:
            return await serpapi_image_search(
                query=query, api_key=self.api_key,
                image_url=image_url, num_results=5,
            )
        except Exception as exc:
            logger.warning("image_search failed: %s", exc)
            return f"(Image search error: {exc})"


# ═══════════════════════════════════════════════════════════════════════════
# Document corpus search (local VDB)
# ═══════════════════════════════════════════════════════════════════════════

class DocSearchTool(SearchTool):
    """Search over a local document corpus (PDF pages → chunks → VDB)."""
    name = "doc_search"
    tag = "doc_search"

    def __init__(self, corpus_dir: str, llm_cfg=None):
        self.corpus_dir = Path(corpus_dir)
        self.llm_cfg = llm_cfg
        self._vdb = None  # lazy init
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

    async def search(self, query: str, top_k: int = 5, **kwargs) -> str:
        self._ensure_index()
        if not self._chunks:
            return "(No document corpus available)"

        results = _bm25_search(query, self._chunks, self._chunk_ids, top_k)
        if not results:
            return "(No matching documents found)"

        lines = []
        for cid, score, text in results:
            lines.append(f"[{cid}] (BM25={score:.1f})\n{text[:500]}")
        return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Media RAG search (audio/video → transcript/keyframes → VDB)
# ═══════════════════════════════════════════════════════════════════════════

class MediaSearchTool(SearchTool):
    """RAG over local audio/video: transcript + keyframe descriptions → VDB."""
    name = "media_search"
    tag = "media_search"

    def __init__(self, corpus_dir: str, llm_cfg=None):
        self.corpus_dir = Path(corpus_dir)
        self.llm_cfg = llm_cfg
        self._vdb = None
        self._chunks: List[str] = []
        self._chunk_ids: List[str] = []

    async def _ensure_index(self):
        """Build media index: transcribe audio/describe video, then BM25."""
        if self._chunks:
            return

        chunks_path = self.corpus_dir / ".media_search_chunks.json"

        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._chunks = data["chunks"]
                self._chunk_ids = data["ids"]
            return

        resources = self.corpus_dir / "resources"
        if not resources.exists():
            return

        print(f"  [MediaSearch] Building index for {self.corpus_dir.name}...")
        chunks = []
        chunk_ids = []

        for f in sorted(resources.iterdir()):
            if f.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"):
                desc = await _describe_video(f, self.llm_cfg)
                if desc:
                    for j, chunk in enumerate(_chunk_text(desc, 256)):
                        chunks.append(chunk)
                        chunk_ids.append(f"vid_{f.stem}_c{j}")
            elif f.suffix.lower() in (".wav", ".mp3", ".flac", ".m4a"):
                transcript = await _transcribe_audio(f, self.llm_cfg)
                if transcript:
                    for j, chunk in enumerate(_chunk_text(transcript, 256)):
                        chunks.append(chunk)
                        chunk_ids.append(f"aud_{f.stem}_c{j}")

        self._chunks = chunks
        self._chunk_ids = chunk_ids

        if chunks:
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump({"chunks": chunks, "ids": chunk_ids}, f, ensure_ascii=False)
            print(f"  [MediaSearch] Indexed {len(chunks)} chunks")

    async def search(self, query: str, top_k: int = 5, **kwargs) -> str:
        await self._ensure_index()
        if not self._chunks:
            return "(No media corpus available)"

        results = _bm25_search(query, self._chunks, self._chunk_ids, top_k)
        if not results:
            return "(No matching media content found)"

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
    "websearch":   ["text_search", "image_search"],    # VL-Web datasets
    "mmreasoning": [],                                 # default fallback
    "docsearch":   ["doc_search"],                     # VL-Doc datasets
    "mediasearch": ["media_search"],                   # Omni datasets
}

# Override task_type for specific dataset categories
DATASET_TOOL_OVERRIDE = {
    # VL-Doc datasets should use doc_search, not default mmreasoning
    "DocBench":        "docsearch",
    "MMLongBench-Doc": "docsearch",
    # Omni datasets: media sent directly to LLM, no search tool needed
    # "OmniVideoBench":  handled via video/audio in prompt
    # "WorldSense":      handled via video/audio in prompt
}


_tool_cache: Dict[str, Dict[str, SearchTool]] = {}


def build_tools(
    task_type: str,
    dataset_name: str = "",
    corpus_dir: str = "",
    llm_cfg=None,
) -> Dict[str, SearchTool]:
    """
    Build the tool set for a given task type / dataset.
    Instances are cached per (dataset_name, corpus_dir) so expensive
    indexes (DocSearch, MediaSearch) are built only once.

    Returns:
        dict mapping tool tag → SearchTool instance
    """
    cache_key = f"{dataset_name}::{corpus_dir}"
    if cache_key in _tool_cache:
        return _tool_cache[cache_key]

    effective_type = DATASET_TOOL_OVERRIDE.get(dataset_name, task_type)
    tool_names = TOOL_PROFILES.get(effective_type, [])

    tools: Dict[str, SearchTool] = {}
    for name in tool_names:
        if name == "text_search":
            tools["text_search"] = TextSearchTool()
        elif name == "image_search":
            tools["image_search"] = ImageSearchTool()
        elif name == "doc_search":
            tools["doc_search"] = DocSearchTool(corpus_dir, llm_cfg)
        elif name == "media_search":
            tools["media_search"] = MediaSearchTool(corpus_dir, llm_cfg)

    _tool_cache[cache_key] = tools
    return tools


def format_available_tools(tools: Dict[str, SearchTool]) -> str:
    """Format available tools for the system prompt."""
    if not tools:
        return ""
    descs = {
        "text_search":  "Text search: <text_search>your query</text_search>",
        "image_search": "Image search: <image_search>describe what to find</image_search>",
        "doc_search":   "Document search: <doc_search>your query</doc_search>",
        "media_search": "Media search: <media_search>your query about audio/video content</media_search>",
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
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
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


async def _describe_video(video_path: Path, llm_cfg) -> str:
    """Extract keyframe descriptions from a video via LLM."""
    from ..llm import async_llm_call_with_images
    import base64

    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed, skipping video: %s", video_path)
        return ""

    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1
        duration = total_frames / fps

        # Sample up to 5 keyframes evenly
        num_frames = min(5, max(1, int(duration / 10)))
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        frames_b64 = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buf = cv2.imencode(".jpg", frame)
                frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        cap.release()

        if not frames_b64:
            return ""

        desc = await async_llm_call_with_images(
            prompt=f"Describe the content of this video ({duration:.0f}s, {num_frames} keyframes). "
                   "Include: main subjects, actions, scene, text overlays, any audio-related visual cues.",
            image_b64_list=frames_b64,
            cfg=llm_cfg,
            max_tokens=512,
        )
        return desc
    except Exception as exc:
        logger.warning("Video description failed for %s: %s", video_path, exc)
        return ""


async def _transcribe_audio(audio_path: Path, llm_cfg) -> str:
    """Transcribe audio via LLM (send as base64 data URL)."""
    from ..llm import async_llm_call
    import base64

    try:
        audio_bytes = audio_path.read_bytes()
        if len(audio_bytes) > 10 * 1024 * 1024:  # 10MB limit
            audio_bytes = audio_bytes[:10 * 1024 * 1024]

        suffix = audio_path.suffix.lower().lstrip(".")
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
                "m4a": "audio/mp4"}.get(suffix, "audio/wav")
        b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Use LLM with audio data URL (supported by Gemini via apiyi)
        transcript = await async_llm_call(
            prompt=f"Transcribe the following audio content. "
                   f"Audio data: [data:{mime};base64,{b64[:100]}...]"
                   f"\nProvide a detailed transcription.",
            cfg=llm_cfg,
            max_tokens=512,
        )
        return transcript
    except Exception as exc:
        logger.warning("Audio transcription failed for %s: %s", audio_path, exc)
        return ""
