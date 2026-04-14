"""
LLM and embedding utilities for HyperSkill.

apiyi integration for HyperSkill:
  - Sync OpenAI client pointing at https://api.apiyi.com/v1
  - API key from api_key_apiyi.txt (project root) or OPENAI_API_KEY env var
  - Generation model : gemini-3.1-flash-lite-preview
  - Embedding model  : gemini-embedding-2-preview  (dim=3072)
  - Audio/image sent as base64 data URLs via chat completions (no Whisper)
  - ThreadPoolExecutor for parallel batch embedding
  - Async wrappers run sync calls in the default executor

Public API
----------
  sync_llm_call(prompt, cfg) -> str
  sync_llm_call_with_images(prompt, image_b64_list, cfg) -> str
  sync_embed_texts(texts, cfg) -> np.ndarray  shape (N, dim)
  sync_embed_image_bytes(image_bytes, cfg) -> np.ndarray  shape (dim,)
  sync_transcribe_audio(audio_bytes, cfg, fmt) -> str
  sync_caption_video_frame(frame_bytes, transcript, cfg) -> str

  async_llm_call(...)              — thin async wrapper
  async_llm_call_with_images(...)  — thin async wrapper
  async_embed_texts(...)           — thin async wrapper
  async_embed_image(...)           — thin async wrapper
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
from openai import OpenAI

from .config import LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def _resolve_api_key(cfg: LLMConfig) -> str:
    if cfg.api_key:
        return cfg.api_key
    # Look for api_key_apiyi.txt in project root (two levels up from this file)
    key_file = Path(__file__).parent.parent / "api_key_apiyi.txt"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return os.environ.get("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# Singleton client cache
# ---------------------------------------------------------------------------

_clients: dict = {}

def _get_client(cfg: LLMConfig) -> OpenAI:
    key = (cfg.base_url, _resolve_api_key(cfg))
    if key not in _clients:
        _clients[key] = OpenAI(
            api_key=key[1],
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )
    return _clients[key]


# ---------------------------------------------------------------------------
# Response-level cache backed by both in-memory LRU and on-disk JSON files
# ---------------------------------------------------------------------------
# Disk cache is essential for reproducibility: two eval runs with the same
# prompts must return byte-identical responses. Gemini at temperature=0 still
# exhibits ~1-3% variation across calls without this layer. Disabled by
# HYPERSKILL_DISABLE_LLM_CACHE=1.

from pathlib import Path as _Path
_LLM_CACHE_DIR = _Path(__file__).resolve().parent.parent / "cache" / "llm"
_llm_cache: dict = {}
_MAX_CACHE = 16384


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{prompt}".encode()).hexdigest()


def _disk_path(ck: str) -> _Path:
    sub = _LLM_CACHE_DIR / ck[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{ck}.txt"


def _cache_get(ck: str) -> Optional[str]:
    if os.environ.get("HYPERSKILL_DISABLE_LLM_CACHE"):
        return None
    if ck in _llm_cache:
        return _llm_cache[ck]
    p = _disk_path(ck)
    if p.exists():
        try:
            val = p.read_text(encoding="utf-8")
            _llm_cache[ck] = val
            return val
        except OSError:
            return None
    return None


def _cache_set(ck: str, val: str) -> None:
    if os.environ.get("HYPERSKILL_DISABLE_LLM_CACHE"):
        return
    if len(_llm_cache) >= _MAX_CACHE:
        for k in list(_llm_cache.keys())[:_MAX_CACHE // 4]:
            del _llm_cache[k]
    _llm_cache[ck] = val
    try:
        _disk_path(ck).write_text(val, encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Sync text completion
# ---------------------------------------------------------------------------

def sync_llm_call(
    prompt: str,
    cfg: LLMConfig,
    system: str = "You are a helpful assistant.",
    use_cache: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """Single-turn text completion via apiyi + Gemini."""
    ck = _cache_key(prompt, cfg.model)
    if use_cache:
        hit = _cache_get(ck)
        if hit is not None:
            return hit

    client = _get_client(cfg)
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens or cfg.max_tokens,
        temperature=temperature if temperature is not None else cfg.temperature,
    )
    text = resp.choices[0].message.content or ""
    if use_cache:
        _cache_set(ck, text)
    return text


# ---------------------------------------------------------------------------
# Sync multimodal completion (image + text)
# ---------------------------------------------------------------------------

def sync_llm_call_with_images(
    prompt: str,
    image_b64_list: List[str],
    cfg: LLMConfig,
    system: str = "You are a helpful assistant.",
    max_tokens: Optional[int] = None,
    audio_bytes: Optional[bytes] = None,
    audio_fmt: str = "wav",
    use_cache: bool = True,
) -> str:
    """
    Chat completion with inline images (base64 JPEG/PNG) and optional audio.
    Sends all modalities in a single request — images via image_url content
    parts, audio via input_audio. Essential for Omni-modal QA (WorldSense)
    where both video frames AND audio are required to answer.
    """
    # Cache key: hash of prompt + system + image hashes + audio hash.
    # This makes vision/audio calls byte-reproducible across eval runs —
    # critical for the Zero-shot vs HyperSkill paired comparison.
    img_fp = "".join(hashlib.sha256(b.encode()).hexdigest()[:8] for b in image_b64_list)
    aud_fp = hashlib.sha256(audio_bytes).hexdigest()[:16] if audio_bytes else ""
    ck = _cache_key(f"{system}\n{prompt}\n[img:{img_fp}][aud:{aud_fp}]", cfg.model)
    if use_cache:
        hit = _cache_get(ck)
        if hit is not None:
            return hit

    client = _get_client(cfg)
    content: list = []
    for b64 in image_b64_list:
        # Detect format from header bytes
        raw = base64.b64decode(b64[:16] + "==")
        mime = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    if audio_bytes:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        content.append({
            "type": "input_audio",
            "input_audio": {"data": audio_b64, "format": audio_fmt},
        })
    content.append({"type": "text", "text": prompt})

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": content},
        ],
        max_tokens=max_tokens or cfg.max_tokens,
        temperature=cfg.temperature,
    )
    text = resp.choices[0].message.content or ""
    if use_cache:
        _cache_set(ck, text)
    return text


# ---------------------------------------------------------------------------
# Sync audio transcription via Gemini (no Whisper endpoint needed)
# ---------------------------------------------------------------------------

_AUDIO_FMT_MIME = {
    "wav":  "audio/wav",
    "mp3":  "audio/mpeg",
    "flac": "audio/flac",
    "m4a":  "audio/mp4",
    "ogg":  "audio/ogg",
}

def sync_transcribe_audio(
    audio_bytes: bytes,
    cfg: LLMConfig,
    fmt: str = "wav",
) -> str:
    """
    Transcribe audio by sending it as a base64 data URL to Gemini.
    Uses the 'input_audio' content part via apiyi chat completions.
    """
    client = _get_client(cfg)
    mime = _AUDIO_FMT_MIME.get(fmt, "audio/wav")
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                {"type": "text", "text": "Please transcribe this audio accurately. Output only the transcript."},
            ]}],
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Audio transcription failed: %s", e)
        return ""


def sync_caption_video_frame(
    frame_bytes: bytes,
    transcript: str,
    cfg: LLMConfig,
    start: float = 0.0,
    end: float = 0.0,
) -> str:
    """Caption a video keyframe with context from its transcript."""
    b64 = base64.b64encode(frame_bytes).decode("utf-8")
    prompt = (
        f"Keyframe from video segment [{start:.1f}s-{end:.1f}s]. "
        f"Transcript: '{transcript[:300]}'. "
        "Describe what you see and relate it to the spoken content."
    )
    return sync_llm_call_with_images(prompt, [b64], cfg, max_tokens=300)


# ---------------------------------------------------------------------------
# Sync batch embedding  (ThreadPoolExecutor, batch=100)
# ---------------------------------------------------------------------------

_EMBED_BATCH = 100
_EMBED_WORKERS = 4


def sync_embed_texts(texts: List[str], cfg: LLMConfig) -> np.ndarray:
    """
    Embed a list of texts.
    Splits into batches of 100 and sends them in parallel threads.
    Returns float32 array of shape (N, embedding_dim).
    """
    if not texts:
        return np.zeros((0, cfg.embedding_dim), dtype=np.float32)

    client = _get_client(cfg)

    def _embed_batch(batch: List[str]) -> List[List[float]]:
        # Replace empty strings to avoid API errors
        batch = [t if t.strip() else " " for t in batch]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=cfg.embedding_model, input=batch)
                if not resp.data:
                    raise ValueError("No embedding data received")
                return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            except Exception as e:
                if attempt < 2:
                    import time; time.sleep(2 ** attempt)
                    continue
                raise

    batches = [texts[i: i + _EMBED_BATCH] for i in range(0, len(texts), _EMBED_BATCH)]
    results: list = [None] * len(batches)
    workers = min(len(batches), _EMBED_WORKERS)

    with ThreadPoolExecutor(max_workers=workers) as exe:
        future_to_idx = {exe.submit(_embed_batch, b): i for i, b in enumerate(batches)}
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()

    flat = [vec for batch_result in results for vec in batch_result]
    return np.array(flat, dtype=np.float32)


def sync_embed_image_bytes(image_bytes: bytes, cfg: LLMConfig) -> np.ndarray:
    """
    Embed an image by first captioning it, then embedding the caption.
    Returns float32 array of shape (embedding_dim,).
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    caption = sync_llm_call_with_images(
        prompt="Briefly describe this image in one or two sentences.",
        image_b64_list=[b64],
        cfg=cfg,
        max_tokens=128,
    )
    vecs = sync_embed_texts([caption], cfg)
    return vecs[0]


# ---------------------------------------------------------------------------
# Async wrappers  (run sync calls in the default thread-pool executor)
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=64)


async def _run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


async def async_llm_call(
    prompt: str,
    cfg: LLMConfig,
    system: str = "You are a helpful assistant.",
    use_cache: bool = True,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    import functools
    fn = functools.partial(
        sync_llm_call, prompt, cfg,
        system=system, use_cache=use_cache,
        max_tokens=max_tokens, temperature=temperature,
    )
    return await _run_in_executor(fn)


async def async_llm_call_with_images(
    prompt: str,
    image_b64_list: List[str],
    cfg: LLMConfig,
    system: str = "You are a helpful assistant.",
    max_tokens: Optional[int] = None,
    audio_bytes: Optional[bytes] = None,
    audio_fmt: str = "wav",
) -> str:
    import functools
    fn = functools.partial(
        sync_llm_call_with_images, prompt, image_b64_list, cfg,
        system=system, max_tokens=max_tokens,
        audio_bytes=audio_bytes, audio_fmt=audio_fmt,
    )
    return await _run_in_executor(fn)


async def async_embed_texts(texts: List[str], cfg: LLMConfig) -> np.ndarray:
    import functools
    fn = functools.partial(sync_embed_texts, texts, cfg)
    return await _run_in_executor(fn)


async def async_embed_image(image_bytes: bytes, cfg: LLMConfig) -> np.ndarray:
    import functools
    fn = functools.partial(sync_embed_image_bytes, image_bytes, cfg)
    return await _run_in_executor(fn)


async def async_transcribe_audio(
    audio_bytes: bytes,
    cfg: LLMConfig,
    fmt: str = "wav",
) -> str:
    import functools
    fn = functools.partial(sync_transcribe_audio, audio_bytes, cfg, fmt)
    return await _run_in_executor(fn)
