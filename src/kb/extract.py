"""PDF -> Markdown extraction with interchangeable backends.

Output: one .md file per PDF, YAML front matter + pages separated by
``<!-- page: N -->`` markers, so chunkers can keep page provenance.

Backends
- pymupdf4llm : fast (seconds per manual), good headings/lists, weaker on merged table cells
- docling     : slower (~1.5 min per manual, ML layout + table models), better tables
- langchain   : PyPDFLoader (pypdf), the LangChain default; plain text, baseline
- mineru      : MinerU 4 local parse server (layout + VLM), Markdown with page markers
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import yaml

from kb import clean
from kb.config import MD_DIR, PAGE_MARKER, PDF_DIR

log = logging.getLogger("kb.extract")

_RE_DOC_CODE = re.compile(r"\b(H\d{5,6}[A-Z]{2}\d)(?=[_\W]|$)")
_RE_LANG = re.compile(r"\(([A-Z]{2})\)\.pdf$")


def pdf_metadata(pdf: Path) -> dict:
    name = pdf.name
    code = _RE_DOC_CODE.search(name)
    lang = _RE_LANG.search(name)
    stem = pdf.stem
    return {
        "source": name,
        "title": stem.split("_", 1)[1].strip() if "_" in stem else stem,
        "doc_code": code.group(1) if code else None,
        "language": lang.group(1).lower() if lang else None,
    }


# ---------------------------------------------------------------- backends
def extract_pymupdf4llm(pdf: Path) -> list[str]:
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(str(pdf), page_chunks=True, show_progress=False)
    return [c["text"] for c in chunks]


def extract_docling(pdf: Path) -> list[str]:
    from docling.document_converter import DocumentConverter

    doc = DocumentConverter().convert(str(pdf)).document
    return [doc.export_to_markdown(page_no=n) for n in sorted(doc.pages)]


def extract_langchain(pdf: Path) -> list[str]:
    """LangChain's stock loader (PyPDFLoader / pypdf): plain text per page, no structure."""
    from langchain_community.document_loaders import PyPDFLoader

    return [d.page_content for d in PyPDFLoader(str(pdf)).load()]


_RE_MINERU_PAGE = re.compile(r"<!-- page (\d+) of (\d+) -->")


def extract_mineru(pdf: Path, tier: str = "standard") -> list[str]:
    """MinerU 4 via its local managed parse server (`mineru server start`, local.mode=managed).

    The CLI writes one Markdown file with ``<!-- page N of M -->`` markers, which we split per page.
    """
    import subprocess
    import tempfile

    import pymupdf

    n_pages = len(pymupdf.open(str(pdf)))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.md"
        cmd = ["mineru", "parse", str(pdf), "--pages", "all", "--tier", tier, "--wait", "3600", "--force", "-o", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"mineru parse failed: {res.stdout[-500:]} {res.stderr[-500:]}")
        text = out.read_text(encoding="utf-8")
    pages = [""] * n_pages
    parts = _RE_MINERU_PAGE.split(text)  # [pre, n, m, body, n, m, body, ...]
    for i in range(1, len(parts), 3):
        idx = int(parts[i]) - 1
        if 0 <= idx < n_pages:
            pages[idx] = parts[i + 2]
    return pages


BACKENDS = {
    "pymupdf4llm": extract_pymupdf4llm,
    "docling": extract_docling,
    "langchain": extract_langchain,
    "mineru": extract_mineru,
}


# ---------------------------------------------------------------- pipeline
def raw_cache_path(pdf: Path, backend: str, out_dir: Path) -> Path:
    return out_dir / "raw" / backend / (pdf.stem + ".pages.json")


def extract_raw(pdf: Path, backend: str, out_dir: Path, force: bool = False) -> list[str]:
    """Backend output per page, cached as JSON so cleaning can be re-run without re-parsing."""
    cache = raw_cache_path(pdf, backend, out_dir)
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))
    pages = BACKENDS[backend](pdf)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


def pdf_to_markdown(pdf: Path, backend: str = "pymupdf4llm", out_dir: Path = MD_DIR, force: bool = False) -> str:
    t0 = time.time()
    raw_pages = extract_raw(pdf, backend, out_dir, force)
    pages = clean.clean_pages(raw_pages)
    meta = pdf_metadata(pdf) | {"extractor": backend, "pages": len(pages)}
    body = "\n\n".join(f"{PAGE_MARKER.format(n=i + 1)}\n\n{p}" for i, p in enumerate(pages))
    log.info("%s: %d pages, %d chars, %.1fs [%s]", pdf.name, len(pages), len(body), time.time() - t0, backend)
    return f"---\n{yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)}---\n\n{body}"


def md_path_for(pdf: Path, backend: str, out_dir: Path = MD_DIR) -> Path:
    return out_dir / backend / (pdf.stem + ".md")


def run(pdfs: list[Path], backend: str, out_dir: Path = MD_DIR, force: bool = False) -> list[Path]:
    written = []
    for pdf in pdfs:
        out = md_path_for(pdf, backend, out_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(pdf_to_markdown(pdf, backend, out_dir, force), encoding="utf-8")
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract PDFs to Markdown")
    ap.add_argument("pdfs", nargs="*", type=Path, help="PDF files (default: every PDF in ./pdf)")
    ap.add_argument("--backend", choices=BACKENDS, default="pymupdf4llm")
    ap.add_argument("--out", type=Path, default=MD_DIR)
    ap.add_argument("--force", action="store_true", help="re-run the backend even if a raw cache exists")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pdfs = args.pdfs or sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit("no PDF found")
    for p in run(pdfs, args.backend, args.out, args.force):
        print(p)


if __name__ == "__main__":
    main()
