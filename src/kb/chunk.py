"""Markdown (with page markers) -> parents (chapter/section) -> child chunks as LangChain Documents.

Chunks are produced *inside* each parent, so a chunk never crosses a section boundary, and each chunk
carries ``parent_id`` plus the parent's chapter/section/page/token metadata.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kb.config import CHUNK_OVERLAP, CHUNK_SIZE, INDEX_FRONT_MATTER, MD_DIR
from kb.parents import RE_PAGE, Parent, build_parents

_RE_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
_RE_HEADING_LINE = re.compile(r"^#{1,4} (.+)$")
_RE_LEADING_NO = re.compile(r"^\d+(?:\.\d+)*\s+")


def read_markdown(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    m = _RE_FRONT.match(text)
    meta = yaml.safe_load(m.group(1)) if m else {}
    return meta, text[m.end():] if m else text


def _blocks(parent: Parent) -> list[tuple[str, str]]:
    """Split the parent's raw text at heading lines -> [(sub_heading, block_text)].
    The parent's own chapter/section headings do not count as sub-headings."""
    own = {parent.chapter_title.lower(), parent.section_title.lower()}
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for line in parent.raw.split("\n"):
        h = _RE_HEADING_LINE.match(line)
        if h:
            title = h.group(1).strip()
            sub = "" if _RE_LEADING_NO.sub("", title).lower() in own else title
            blocks.append((sub, [line]))
        else:
            blocks[-1][1].append(line)
    return [(s, "\n".join(l)) for s, l in blocks if "\n".join(l).strip()]


def chunk_parent(parent: Parent, meta: dict, splitter: RecursiveCharacterTextSplitter, start_index: int) -> list[Document]:
    docs: list[Document] = []
    page = parent.page_start
    stem = Path(parent.source).stem
    for sub, block in _blocks(parent):
        crumb = parent.title + (f" > {sub}" if sub else "")
        for piece in splitter.split_text(block):
            markers = RE_PAGE.findall(piece)
            page_start = page
            if markers:
                page = int(markers[-1])
                if piece.lstrip().startswith("<!-- page:"):
                    page_start = int(markers[0])
            text = RE_PAGE.sub("", piece).strip()
            if len(text) < 40:
                continue
            docs.append(
                Document(
                    page_content=f"{crumb}\n\n{text}",
                    metadata={
                        "source": parent.source,
                        "doc_code": parent.doc_code,
                        "language": parent.language,
                        "extractor": meta.get("extractor") or "",
                        "page_start": page_start,
                        "page_end": page,
                        "section": crumb,
                        "chunk_id": f"{stem}#{start_index + len(docs)}",
                        # parent document
                        "parent_id": parent.parent_id,
                        "parent_title": parent.title,
                        "parent_tokens": parent.tokens,
                        "parent_page_start": parent.page_start,
                        "parent_page_end": parent.page_end,
                        "chapter_no": parent.chapter_no,
                        "chapter_title": parent.chapter_title,
                        "section_no": parent.section_no,
                        "section_title": parent.section_title,
                        "chunk_in_parent": len([d for d in docs if d.metadata["parent_id"] == parent.parent_id]),
                    },
                )
            )
    return docs


def process_markdown(path: Path, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> tuple[list[Document], list[Parent]]:
    meta, body = read_markdown(path)
    parents = build_parents(meta, body)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs: list[Document] = []
    for parent in parents:
        if parent.chapter_no == "0" and not INDEX_FRONT_MATTER:
            continue
        docs.extend(chunk_parent(parent, meta, splitter, len(docs)))
    return docs, parents


def chunk_markdown(path: Path, **kw) -> list[Document]:
    return process_markdown(path, **kw)[0]


def build_corpus(extractor: str, md_dir: Path = MD_DIR, **kw) -> tuple[list[Document], list[Parent]]:
    """Chunks and parents for every Markdown file of an extractor."""
    docs: list[Document] = []
    parents: list[Parent] = []
    for path in sorted((md_dir / extractor).glob("*.md")):
        d, p = process_markdown(path, **kw)
        docs.extend(d)
        parents.extend(p)
    return docs, parents


def load_corpus(extractor: str, md_dir: Path = MD_DIR, **kw) -> list[Document]:
    return build_corpus(extractor, md_dir, **kw)[0]


if __name__ == "__main__":
    import sys

    docs, parents = process_markdown(Path(sys.argv[1]))
    for p in parents:
        print(f"{p.parent_id}  p{p.page_start}-{p.page_end}  {p.tokens:>5} tok  {p.title}")
    print(len(parents), "parents,", len(docs), "chunks")
