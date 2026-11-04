"""Build the doc_search BM25 index of DocBench and MMLongBench-Doc ($0, local CPU only).

The index <data>/VL-Doc/<dataset>/.doc_search_chunks.json holds the text of every PDF page of
the dataset's resources/ (PyMuPDF text layer, 512-word chunks). PDFs without any text layer
(28 in MMLongBench-Doc, 0 in DocBench) are first OCR'd with RapidOCR into the sidecars
<data>/VL-Doc/<dataset>/.ocr_pages/<stem>.json, which the index builder reads instead.

    pip install rapidocr-onnxruntime==1.2.3        # only needed for --ocr
    python scripts/build_doc_index.py --dataset MMLongBench-Doc --ocr
    python scripts/build_doc_index.py --dataset DocBench

Without this step the index is built on the first doc_search call, but without the OCR
sidecars (image-only PDFs then have no text). The index of the reported runs has sha256
EXPECTED_SHA256 below; the script prints the sha256 of the index it built.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hyperskill.agents.tools import DocSearchTool, _extract_pdf_text  # noqa: E402
from hyperskill.datasets.loaders import DATASET_KEYS, DATASET_ROOT  # noqa: E402

EXPECTED_SHA256 = {
    "DocBench": "57ce2e736a256569dc63f836e61c3a38b4e0167bb86b0a4677f80ea3d2e2f308",
    "MMLongBench-Doc": "66da1e68c6d893383ea650dbd696a1f124b6d30d89c1c59804b89184870df734",
}


def ocr_pdf(args):
    """Page texts of one PDF: each page rendered at 2x, RapidOCR lines joined by newlines."""
    pdf, out = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import pymupdf
    import onnxruntime as ort
    import rapidocr_onnxruntime.utils as U

    def opts():                      # 2 intra-op threads per session, as for the reported index
        o = ort.SessionOptions()
        o.intra_op_num_threads, o.inter_op_num_threads = 2, 1
        return o
    U.SessionOptions = opts
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()
    doc = pymupdf.open(pdf)
    pages = []
    t0 = time.time()
    for page in doc:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        res, _ = eng(pix.tobytes("png"))
        pages.append("\n".join(r[1] for r in (res or []) if r and r[1].strip()))
    doc.close()
    tmp = Path(out).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pages, ensure_ascii=False))
    os.replace(tmp, out)
    return Path(pdf).name, len(pages), round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(EXPECTED_SHA256))
    ap.add_argument("--ocr", action="store_true", help="OCR the PDFs without a text layer first (RapidOCR)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="rebuild the index if it exists")
    a = ap.parse_args()

    corpus = DATASET_ROOT / DATASET_KEYS[a.dataset]["folder"]
    resources = corpus / "resources"
    if not resources.is_dir():
        sys.exit(f"{resources} not found (download the data first, see README)")
    index = corpus / ".doc_search_chunks.json"

    if a.ocr:
        no_text = [p for p in sorted(resources.iterdir())
                   if p.suffix.lower() == ".pdf" and not any(t.strip() for t in _extract_pdf_text(p))]
        od = corpus / ".ocr_pages"
        od.mkdir(exist_ok=True)
        todo = [(str(p), str(od / f"{p.stem}.json")) for p in no_text if not (od / f"{p.stem}.json").exists()]
        print(f"{len(no_text)} PDFs without a text layer, {len(todo)} to OCR", flush=True)
        if todo:
            with Pool(a.workers) as pool:
                for name, n_pages, sec in pool.imap_unordered(ocr_pdf, todo):
                    print(f"  OCR {name}: {n_pages} pages, {sec}s", flush=True)

    if index.exists():
        if not a.force:
            sys.exit(f"{index} exists (use --force to rebuild)")
        index.unlink()
    tool = DocSearchTool(str(corpus))
    tool._ensure_index()
    sha = hashlib.sha256(index.read_bytes()).hexdigest()
    ok = sha == EXPECTED_SHA256[a.dataset]
    print(f"{index}: {len(tool._chunks)} chunks, sha256 {sha} "
          f"({'matches' if ok else 'differs from'} the index of the reported runs)")


if __name__ == "__main__":
    main()
