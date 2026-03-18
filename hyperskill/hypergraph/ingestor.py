"""
Omni-modal data ingestor for HyperSkill.

Converts raw multimodal data into TextChunk records for hypergraph insertion.
All modalities are processed via apiyi + Gemini (gemini-3.1-flash-lite-preview),
All API calls go through apiyi using the sync OpenAI client.

Supported modalities
--------------------
  text   : plain string, token-chunked
  image  : bytes (JPEG/PNG) → Gemini caption → text chunk
  pdf    : bytes → per-page text + rendered image caption
  audio  : bytes (WAV/MP3/…) → Gemini transcription → text chunks
           segments saved to expr_dir via soundfile
  video  : file path → ffmpeg keyframes + audio → Gemini caption/transcript
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import LLMConfig
from ..llm import (
    sync_llm_call_with_images,
    sync_transcribe_audio,
    sync_caption_video_frame,
)

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A unit of text derived from any modality."""
    chunk_id: str
    content: str
    modality: str           # text | image | audio | video | pdf
    source: str
    advantage: float = 0.0
    media_ref: Optional[str] = None   # base64-encoded media or file path


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

async def ingest_text(
    text: str,
    source: str,
    advantage: float,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[TextChunk]:
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i, start in enumerate(range(0, len(words), step)):
        content = " ".join(words[start: start + chunk_size])
        if not content.strip():
            continue
        chunks.append(TextChunk(
            chunk_id=_make_id(source, "text", i),
            content=content,
            modality="text",
            source=source,
            advantage=advantage,
        ))
    return chunks


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

def _sync_ingest_image(
    image_bytes: bytes,
    source: str,
    advantage: float,
    llm_cfg: LLMConfig,
    caption_prompt: str = "Describe this image in detail, focusing on what reasoning or knowledge it demonstrates.",
    idx: int = 0,
) -> TextChunk:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    caption = sync_llm_call_with_images(
        prompt=caption_prompt,
        image_b64_list=[b64],
        cfg=llm_cfg,
        max_tokens=256,
    )
    return TextChunk(
        chunk_id=_make_id(source, "image", idx),
        content=caption,
        modality="image",
        source=source,
        advantage=advantage,
        media_ref=b64,
    )


async def ingest_image(
    image_bytes: bytes,
    source: str,
    advantage: float,
    llm_cfg: LLMConfig,
    idx: int = 0,
) -> TextChunk:
    import asyncio, functools
    fn = functools.partial(_sync_ingest_image, image_bytes, source, advantage, llm_cfg, idx=idx)
    return await asyncio.get_event_loop().run_in_executor(None, fn)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

async def ingest_pdf(
    pdf_bytes: bytes,
    source: str,
    advantage: float,
    llm_cfg: LLMConfig,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[TextChunk]:
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required: pip install PyMuPDF")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_chunks: List[TextChunk] = []
    for page_num, page in enumerate(doc):
        # Text chunks from page
        text = page.get_text("text").strip()
        if text:
            page_chunks = await ingest_text(text, source, advantage, chunk_size, overlap)
            for c in page_chunks:
                c.modality = "pdf"
                c.chunk_id = _make_id(source, f"pdf_p{page_num}", int(c.chunk_id[-4:], 16) % 1000)
            all_chunks.extend(page_chunks)
        # Caption the rendered page image
        pix = page.get_pixmap(dpi=72)
        img_bytes = pix.tobytes("jpeg")
        img_chunk = await ingest_image(img_bytes, source, advantage, llm_cfg, idx=page_num)
        img_chunk.modality = "pdf"
        img_chunk.chunk_id = _make_id(source, f"pdf_img_p{page_num}", 0)
        all_chunks.append(img_chunk)
    doc.close()
    return all_chunks


# ---------------------------------------------------------------------------
# Audio  (Gemini transcription via apiyi)
# ---------------------------------------------------------------------------

async def ingest_audio(
    audio_bytes: bytes,
    source: str,
    advantage: float,
    llm_cfg: LLMConfig,
    segment_duration: float = 30.0,
    expr_dir: str = "expr",
    idx: int = 0,
) -> List[TextChunk]:
    """
    Segment audio with soundfile, transcribe each segment via Gemini.
    Segment WAV files are written to expr_dir.
    """
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        raise ImportError("soundfile required: pip install soundfile")

    import asyncio, functools

    buf = io.BytesIO(audio_bytes)
    try:
        data, samplerate = sf.read(buf, always_2d=False)
    except Exception as e:
        logger.error("Audio read failed: %s", e)
        return [TextChunk(
            chunk_id=_make_id(source, "audio", idx),
            content=f"[Audio ingestion failed: {e}]",
            modality="audio",
            source=source,
            advantage=advantage,
        )]

    os.makedirs(expr_dir, exist_ok=True)
    samples_per_seg = int(samplerate * segment_duration)
    segs: List[Tuple[float, float]] = []
    start = 0.0
    while start < len(data) / samplerate:
        end = min(start + segment_duration, len(data) / samplerate)
        segs.append((start, end))
        start += segment_duration

    # Extract segment WAV bytes and transcribe in threadpool
    chunks: List[TextChunk] = []

    def _process_seg(seg_i: int, seg_start: float, seg_end: float) -> TextChunk:
        s = int(seg_start * samplerate)
        e = int(min(seg_end * samplerate, len(data)))
        segment = data[s:e]
        # Write to temp WAV
        buf_out = io.BytesIO()
        sf.write(buf_out, segment, samplerate, format="WAV")
        seg_bytes = buf_out.getvalue()
        transcript = sync_transcribe_audio(seg_bytes, llm_cfg, fmt="wav")
        return TextChunk(
            chunk_id=_make_id(source, f"audio_{idx}", seg_i),
            content=transcript or f"[{seg_start:.1f}s-{seg_end:.1f}s]",
            modality="audio",
            source=source,
            advantage=advantage,
        )

    loop = asyncio.get_event_loop()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
        futs = [
            loop.run_in_executor(exe, _process_seg, i, s, e)
            for i, (s, e) in enumerate(segs)
        ]
        results = await asyncio.gather(*futs)
    return list(results)


# ---------------------------------------------------------------------------
# Video  (ffmpeg keyframes + Gemini captions)
# ---------------------------------------------------------------------------

async def ingest_video(
    video_path: str,
    source: str,
    advantage: float,
    llm_cfg: LLMConfig,
    segment_duration: float = 30.0,
    expr_dir: str = "expr",
) -> List[TextChunk]:
    """
    Segment video with ffmpeg, extract audio + keyframe per segment,
    transcribe audio and caption keyframe via Gemini.
    """
    import asyncio, concurrent.futures, functools

    os.makedirs(expr_dir, exist_ok=True)

    # Get duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, check=True,
        )
        duration = float(result.stdout.strip())
    except Exception as e:
        logger.error("ffprobe failed for %s: %s", video_path, e)
        return []

    segs: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + segment_duration, duration)
        segs.append((start, end))
        start += segment_duration

    def _process_seg(seg_i: int, seg_start: float, seg_end: float) -> TextChunk:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        # Extract audio segment to WAV
        audio_path = os.path.join(expr_dir, f"{stem}_{seg_start:.1f}s_{seg_end:.1f}s.wav")
        frame_path = os.path.join(expr_dir, f"{stem}_{seg_start:.1f}s_{seg_end:.1f}s.jpg")

        transcript = ""
        # Audio extraction
        try:
            subprocess.run(
                ["ffmpeg", "-y",
                 "-ss", str(seg_start), "-t", str(seg_end - seg_start),
                 "-i", video_path,
                 "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                 audio_path],
                capture_output=True, check=True,
            )
            with open(audio_path, "rb") as f:
                transcript = sync_transcribe_audio(f.read(), llm_cfg, fmt="wav")
        except Exception as e:
            logger.warning("Audio extract/transcribe failed [%.1fs]: %s", seg_start, e)

        # Keyframe extraction
        description = f"[{seg_start:.1f}s-{seg_end:.1f}s] {transcript[:200]}"
        try:
            mid = (seg_start + seg_end) / 2
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(mid), "-i", video_path,
                 "-vframes", "1", "-q:v", "2", frame_path],
                capture_output=True, check=True,
            )
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
            description = sync_caption_video_frame(frame_bytes, transcript, llm_cfg, seg_start, seg_end)
        except Exception as e:
            logger.warning("Keyframe caption failed [%.1fs]: %s", seg_start, e)

        return TextChunk(
            chunk_id=_make_id(source, "video", seg_i),
            content=description,
            modality="video",
            source=source,
            advantage=advantage,
        )

    loop = asyncio.get_event_loop()
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=4) as exe:
        futs = [
            loop.run_in_executor(exe, _process_seg, i, s, e)
            for i, (s, e) in enumerate(segs)
        ]
        results = await asyncio.gather(*futs)
    return list(results)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

async def ingest_modality(
    modality: str,
    data: Any,
    source: str,
    advantage: float,
    llm_cfg: LLMConfig,
    chunk_size: int = 512,
    overlap: int = 64,
    expr_dir: str = "expr",
) -> List[TextChunk]:
    """
    Dispatch to the appropriate ingestor.

    Args:
        modality : "text" | "image" | "pdf" | "audio" | "video"
        data     : str (text), bytes (image/pdf/audio), or path str (video)
    """
    if modality == "text":
        return await ingest_text(data, source, advantage, chunk_size, overlap)
    elif modality == "image":
        return [await ingest_image(data, source, advantage, llm_cfg)]
    elif modality == "pdf":
        return await ingest_pdf(data, source, advantage, llm_cfg, chunk_size, overlap)
    elif modality == "audio":
        return await ingest_audio(data, source, advantage, llm_cfg, expr_dir=expr_dir)
    elif modality == "video":
        return await ingest_video(data, source, advantage, llm_cfg, expr_dir=expr_dir)
    else:
        raise ValueError(f"Unknown modality: {modality}")


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------

def _make_id(source: str, modality: str, idx: int) -> str:
    raw = f"{source}_{modality}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()
