"""Markdown (with page markers) -> LangChain Documents with section + page metadata."""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from kb.config import CHUNK_OVERLAP, CHUNK_SIZE, MD_DIR

_RE_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
_RE_PAGE = re.compile(r"<!-- page: (\d+) -->\n?")
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def read_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = _RE_FRONT.match(text)
    meta = yaml.safe_load(m.group(1)) if m else {}
    return meta, text[m.end():] if m else text


def _breadcrumb(meta: dict, md: dict) -> str:
    parts = [meta.get("title") or Path(meta["source"]).stem]
    parts += [md[k] for k in ("h1", "h2", "h3", "h4") if md.get(k)]
    return " > ".join(parts)


def chunk_markdown(path: Path, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Document]:
    meta, body = read_markdown(path)
    sections = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False).split_text(body)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs: list[Document] = []
    page = 1
    for sec in sections:
        for i, piece in enumerate(splitter.split_text(sec.page_content)):
            markers = _RE_PAGE.findall(piece)
            page_start = page
            if markers:
                page = int(markers[-1])
                # a chunk that *starts* with a marker belongs entirely to the new page
                if piece.lstrip().startswith("<!-- page:"):
                    page_start = int(markers[0])
            text = _RE_PAGE.sub("", piece).strip()
            if len(text) < 40:  # marker-only or empty leftovers
                continue
            crumb = _breadcrumb(meta, sec.metadata)
            docs.append(
                Document(
                    page_content=f"{crumb}\n\n{text}",
                    metadata={
                        "source": meta.get("source"),
                        "doc_code": meta.get("doc_code"),
                        "language": meta.get("language"),
                        "extractor": meta.get("extractor"),
                        "page_start": page_start,
                        "page_end": page,
                        "section": crumb,
                        "chunk_id": f"{Path(meta['source']).stem}#{len(docs)}",
                    },
                )
            )
    return docs


def load_corpus(extractor: str, md_dir: Path = MD_DIR, **kw) -> list[Document]:
    docs: list[Document] = []
    for path in sorted((md_dir / extractor).glob("*.md")):
        docs.extend(chunk_markdown(path, **kw))
    return docs


if __name__ == "__main__":
    import sys

    for d in chunk_markdown(Path(sys.argv[1]))[:5]:
        print(d.metadata, "\n", d.page_content[:300], "\n", "-" * 60)
