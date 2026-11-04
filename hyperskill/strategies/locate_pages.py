"""
Locate pages, then answer (`hypergraph.locate_pages`, doc QA, evaluation only; 0 = off).

Before the first model call, BM25 over the question's own PDF with the question
(plus every page / slide the question names) ranks its pages; the text of the
top n pages is put after the question, with the rule to answer from them,
search again if they lack the answer, and answer "Not answerable" only if the
searches do not find it. Same text layer and BM25 index as the doc_search tool;
no page images, no API call.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

_STOP = set("""a an the of in on at to for from by with and or is are was were be been being what which who
whom whose when where why how this that these those it its as do does did any all there their them they
he she his her than then into about according document question here pdf page pages slide slides please
give list many much number""".split())
_PREFIX = re.compile(r"^\s*Here is a question from .*? document\s*:\s*", re.I | re.S)


def doc_query(question: str) -> str:
    q = _PREFIX.sub("", question or "")
    toks = []
    for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.%$']*", q.lower()):
        t = t.strip(".'")
        if t and t not in _STOP and t not in toks:
            toks.append(t)
    return " ".join(toks)


def named_pages(question: str) -> List[int]:
    """0-based page indices of the pages / slides a question names ("page 17", "slide 3")."""
    q = _PREFIX.sub("", question or "")
    out = []
    for m in re.finditer(r"\b(?:page|slide|p\.)\s*(\d{1,4})\b", q, re.I):
        i = int(m.group(1)) - 1
        if i >= 0 and i not in out:
            out.append(i)
    return out


def locate_pages(tool, resource_paths: List[str], question: str, n_pages: int,
                 words_per_page: int = 700) -> Dict[str, Any]:
    """Top-n pages of the question's own PDF(s): {"pages": [(stem, page_idx, score, text)],
    "query": str, "no_text": bool}. Pure function of the BM25 index (no API call)."""
    from ..agents.tools import _bm25_search
    tool._ensure_index()
    stems = [Path(p).stem for p in resource_paths]
    keep = [i for i, cid in enumerate(tool._chunk_ids) if any(cid.startswith(f"doc_{s}_") for s in stems)]
    if not keep:
        return {"pages": [], "query": "", "no_text": True}
    chunks = [tool._chunks[i] for i in keep]
    ids = [tool._chunk_ids[i] for i in keep]
    page_of = {}
    for cid in ids:
        m = re.match(r"doc_(.+)_p(\d+)_c(\d+)$", cid)
        if m:
            page_of[cid] = (m.group(1), int(m.group(2)))
    text_of: Dict[Tuple[str, int], List[Tuple[int, str]]] = {}
    for cid, ch in zip(ids, chunks):
        if cid in page_of:
            text_of.setdefault(page_of[cid], []).append((int(cid.rsplit("_c", 1)[1]), ch))
    query = doc_query(question)
    ranked: List[Tuple[Tuple[str, int], float]] = []
    seen = set()
    for k in named_pages(question):                         # explicitly named pages first
        for s in dict.fromkeys(page_of[c][0] for c in ids if c in page_of):
            if (s, k) in text_of and (s, k) not in seen:
                ranked.append(((s, k), float("inf"))); seen.add((s, k))
    for cid, score, _ in (_bm25_search(query, chunks, ids, top_k=len(chunks)) if query else []):
        p = page_of.get(cid)
        if p is not None and p not in seen:
            ranked.append((p, float(score))); seen.add(p)
    if not ranked:                                            # no term matches: the first text pages
        ranked = [(p, 0.0) for p in sorted(text_of, key=lambda x: (stems.index(x[0]) if x[0] in stems else 0, x[1]))]
    pages = []
    for (s, k), score in ranked[:n_pages]:
        words = " ".join(t for _, t in sorted(text_of[(s, k)])).split()
        pages.append((s, k, score, " ".join(words[:words_per_page])))
    return {"pages": pages, "query": query, "no_text": False}


def pages_context(loc: Dict[str, Any], resource_paths: List[str]) -> str:
    names = ", ".join(sorted(Path(p).name for p in resource_paths))
    if loc["no_text"]:
        return (f"Document search could not read any text in {names} (no text layer). Search it with "
                f"<doc_search>...</doc_search> if useful; answer \"Not answerable\" only if the "
                f"information cannot be found.")
    multi = len({s for s, *_ in loc["pages"]}) > 1
    blocks = [f"[{s + '.pdf ' if multi else ''}page {k + 1}]\n{t}" for s, k, _, t in loc["pages"]]
    return (f"Candidate pages of {names}, located by document search for this question (text extracted "
            f"from the PDF):\n\n" + "\n\n".join(blocks) + "\n\n"
            "Answer from these pages. If they do not contain the answer, search the document again with "
            "<doc_search>...</doc_search>. Answer \"Not answerable\" only if the information is still not "
            "found after searching. Put the final answer inside <answer>...</answer>.")


_LOC_MEMO: Dict[Tuple[str, Tuple[str, ...], int], Dict[str, Any]] = {}


def located(tool, item, n_pages: int) -> Dict[str, Any]:
    key = (item.question, tuple(item.resource_paths), n_pages)
    if key not in _LOC_MEMO:
        _LOC_MEMO[key] = locate_pages(tool, item.resource_paths, item.question, n_pages)
    return _LOC_MEMO[key]


def _doc_tool(item):
    from ..agents.tools import build_tools
    return build_tools(task_type=item.task_type, dataset_name=item.source,
                       corpus_dir=item.dataset_dir).get("doc_search")


def lp_suffix(trainer, item) -> str:
    """Page block after the question ("" when off or the item has no doc_search)."""
    n = trainer.cfg.hypergraph.locate_pages
    if n <= 0 or not item.resource_paths:
        return ""
    tool = _doc_tool(item)
    if tool is None:
        return ""
    return pages_context(located(tool, item, n), item.resource_paths)


def lp_record(trainer, item):
    """Record of the located pages (None when off or the item has no doc_search)."""
    n = trainer.cfg.hypergraph.locate_pages
    if n <= 0 or not item.resource_paths:
        return None
    tool = _doc_tool(item)
    if tool is None:
        return None
    loc = located(tool, item, n)
    return {"query": loc["query"], "no_text": loc["no_text"],
            "pages": [[s, k + 1, (None if sc == float("inf") else round(sc, 3))] for s, k, sc, _ in loc["pages"]]}
