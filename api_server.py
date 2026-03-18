"""
HyperSkill Skill Library API Server (FastAPI) — optional HTTP interface.

The skill library runs embedded during training/eval by default.
This server is only needed if you want to expose the library over HTTP
for external tools or distributed setups.

Endpoints:
  POST /memory  — store a trajectory experience in the skill library
  POST /search  — query the skill library for relevant skills

Run with:
    python api_server.py --dataset dataset/Text/AIME2024 --port 8102
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
from typing import List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from hyperskill.config import load_config, HyperSkillConfig
from hyperskill.skill.library import SkillLibrary

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HyperSkill Skill Library API",
    description="Omni-modal weighted hypergraph skill library server.",
    version="1.0.0",
)

_library: SkillLibrary | None = None
_cfg: HyperSkillConfig | None = None
_lock = asyncio.Lock()


def get_library() -> SkillLibrary:
    if _library is None:
        raise RuntimeError("SkillLibrary not initialised. Call /init or start with --config.")
    return _library


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MemoryRequest(BaseModel):
    """
    Store a trajectory experience.

    queries format:
      - N experience JSON strings:
          {"trajectory": "...", "advantage": 0.8, "source": "FVQA",
           "modality": "vl", "hop_type": "single"}
      - Last element: config JSON string (optional)
    """
    queries: List[str]


class SearchRequest(BaseModel):
    """
    Query the skill library.

    queries format:
      List of JSON strings:
        {"query": "...", "query_type": "text"|"image", "source": "FVQA"}
    """
    queries: List[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/memory")
async def memory(request: MemoryRequest):
    """Insert trajectory experiences into the skill library."""
    async with _lock:
        lib = get_library()
        queries = request.queries

        # Last element might be a config dict
        experiences = []
        for q in queries:
            try:
                item = json.loads(q)
                if "trajectory" in item:
                    experiences.append(item)
            except json.JSONDecodeError:
                pass

        if not experiences:
            return {"status": "no experiences provided"}

        loop = asyncio.get_running_loop()
        inserted = await loop.run_in_executor(
            None,
            lambda: asyncio.run(_batch_evolve(lib, experiences)),
        )
        lib.save()
        return {"status": "ok", "inserted": inserted}


async def _batch_evolve(lib: SkillLibrary, experiences: list) -> int:
    return await lib.batch_evolve(experiences)


@app.post("/search")
async def search(request: SearchRequest):
    """Search the skill library for relevant skills."""
    async with _lock:
        lib = get_library()
        query_items = []
        for q in request.queries:
            try:
                query_items.append(json.loads(q))
            except json.JSONDecodeError:
                query_items.append({"query": q, "query_type": "text"})

        results = []
        for item in query_items:
            query_text = item.get("query", "")
            query_type = item.get("query_type", "text")
            source_filter = item.get("source", None)

            image_bytes = None
            if query_type == "image":
                import base64
                try:
                    image_bytes = base64.b64decode(query_text)
                    query_text = ""
                except Exception:
                    pass

            skills = await lib.retrieve(
                query=query_text or "multimodal query",
                top_k=10,
                image_bytes=image_bytes,
                source_filter=source_filter,
            )
            skill_context = "\n".join([
                f"[Skill] {s.name}: {s.description}" for s in skills
            ])
            results.append(json.dumps({"Here is the experience": skill_context}))

        return results


@app.get("/stats")
async def stats():
    """Return skill library statistics."""
    lib = get_library()
    return lib.stats()


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global _library, _cfg
    # Config is set from CLI before uvicorn starts
    if _cfg is not None and _library is None:
        _library = SkillLibrary(_cfg.hypergraph, _cfg.llm)
        print(f"[HyperSkill API] Skill library loaded from: {_cfg.hypergraph.working_dir}")
        print(f"[HyperSkill API] Stats: {_library.stats()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperSkill Skill Library API Server")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/text_single_hop.yaml)"
    )
    parser.add_argument(
        "--port", type=int, default=8102,
        help="Port to run the server on (default: 8102)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    _cfg = cfg

    print(f"[HyperSkill API] Starting server on {args.host}:{args.port}")
    print(f"[HyperSkill API] Skill library at: {cfg.hypergraph.working_dir}")

    uvicorn.run(app, host=args.host, port=args.port)
