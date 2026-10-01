"""Structure-aware parent documents built from the extracted Markdown.

A parent is a chapter (when small) or a section of a large chapter. Parents are *not* written as
separate files and are not read back through offsets: their text is stored in a docstore keyed by
``parent_id`` (a payload-only Qdrant collection, see kb.hybrid). ``char_start`` / ``char_end`` are
kept only as provenance into the Markdown body.

Heading styles handled:
- "N TITLE" chapters and "N.M TITLE" sections (most manuals)
- "N.0 TITLE" chapters (Fulvia)
- unnumbered manuals: UPPERCASE headings are chapters
- numbered legend items such as "2 LED" are rejected because chapter numbers must be in sequence
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from kb.config import PARENT_MAX_TOKENS, PARENT_MIN_TOKENS, PARENT_TOKENIZER

RE_PAGE = re.compile(r"<!-- page: (\d+) -->\n?")
_RE_PAGE_LINE = re.compile(r"^<!-- page: (\d+) -->$")
_RE_NUM = re.compile(r"^#{1,4} (\d+)((?:\.\d+)*) (.+)$")
_RE_ANY = re.compile(r"^#{1,4} (.+)$")
FRONT_TITLE = "Copertina e indice"


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(PARENT_TOKENIZER)


def count_tokens(text: str) -> int:
    return len(_tokenizer()(text, add_special_tokens=False, truncation=False)["input_ids"])


def strip_markers(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", RE_PAGE.sub("", text)).strip()


@dataclass
class Unit:
    """A chapter heading block (level 1) or a section (level 2), as a line span of the body."""
    level: int
    number: str
    title: str
    start: int  # first line index (inclusive)
    end: int = 0  # last line index (exclusive)


@dataclass
class Parent:
    parent_id: str
    doc_code: str
    source: str
    doc_title: str
    language: str
    chapter_no: str
    chapter_title: str
    section_no: str
    section_title: str
    part: int  # 0 = not split, otherwise 1..n
    parts: int
    title: str  # breadcrumb: doc title > chapter > section
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    tokens: int
    text: str  # clean text for the LLM (page markers removed)
    raw: str = field(repr=False, default="")  # body slice with page markers, used for chunking

    def payload(self) -> dict:
        d = asdict(self)
        d.pop("raw")
        return d


def outline(lines: list[str]) -> list[Unit]:
    numbered = sum(1 for l in lines if _RE_NUM.match(l))
    units: list[Unit] = []
    cur = Unit(1, "0", FRONT_TITLE, 0)
    chap, page, last_content = 0, 1, -1
    for i, line in enumerate(lines):
        m = _RE_PAGE_LINE.match(line)
        if m:
            page = int(m.group(1))
            continue
        if not line.strip():
            continue
        new: Unit | None = None
        h = _RE_NUM.match(line)
        if h:
            major, rest, title = int(h.group(1)), h.group(2), h.group(3).strip()
            if rest in ("", ".0"):
                if chap < major <= chap + 3 and title.upper() == title:
                    chap = major
                    new = Unit(1, str(major), title, 0)
            elif rest.count(".") == 1 and major in (chap, chap + 1):
                chap = major
                new = Unit(2, f"{major}{rest}", title, 0)
        elif numbered < 5:
            a = _RE_ANY.match(line)
            if a and a.group(1).upper() == a.group(1) and len(a.group(1)) > 6 and page > 2:
                new = Unit(1, "", a.group(1).strip(), 0)
        if new:
            boundary = last_content + 1  # page markers / blanks right before a heading go with it
            cur.end = boundary
            units.append(cur)
            new.start = boundary
            cur = new
        last_content = i
    cur.end = len(lines)
    units.append(cur)
    return [u for u in units if u.end > u.start]


def _split_span(lines: list[str], start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Split an oversize line span into contiguous parts <= limit tokens, cutting at inner headings,
    then at blank lines, then at single lines (e.g. a long table of contents)."""
    is_content = lambda i: bool(lines[i].strip()) and not _RE_PAGE_LINE.match(lines[i])
    size = lambda a, b: count_tokens("\n".join(lines[a:b]))

    def pack(a: int, b: int, points: list[int]) -> list[tuple[int, int]]:
        bounds = [a] + [x for x in points if a < x < b] + [b]
        parts: list[tuple[int, int]] = []
        acc_start, acc_tok = a, 0
        for x, y in zip(bounds, bounds[1:]):
            t = size(x, y)
            if acc_tok and acc_tok + t > limit:
                parts.append((acc_start, x))
                acc_start, acc_tok = x, 0
            acc_tok += t
        parts.append((acc_start, b))
        return parts

    def refine(spans: list[tuple[int, int]], points_fn) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for a, b in spans:
            out += pack(a, b, points_fn(a, b)) if size(a, b) > limit else [(a, b)]
        return out

    first = next((i for i in range(start, end) if is_content(i)), start)
    spans = [(start, end)]
    # 1) inner headings (never the span's own first heading, so no empty leading part)
    spans = refine(spans, lambda a, b: [i for i in range(max(a, first) + 1, b) if _RE_ANY.match(lines[i])])
    # 2) blank lines   3) every line
    spans = refine(spans, lambda a, b: [i + 1 for i in range(a, b - 1) if not lines[i].strip()])
    spans = refine(spans, lambda a, b: list(range(a + 1, b)))
    return [(a, b) for a, b in spans if any(is_content(i) for i in range(a, b))]


def _join_sections(first: Unit | None, second: Unit | None) -> Unit | None:
    """Label for a parent made of two adjacent sections (a tiny one merged into its sibling)."""
    if first is None or second is None:
        return first or second
    return Unit(2, f"{first.number}, {second.number}", f"{first.title} / {second.title}", first.start, second.end)


def build_parents(meta: dict, body: str, max_tokens: int = PARENT_MAX_TOKENS, min_tokens: int = PARENT_MIN_TOKENS) -> list[Parent]:
    lines = body.split("\n")
    offsets = [0]
    for l in lines:
        offsets.append(offsets[-1] + len(l) + 1)
    page_at, page = [], 1
    for l in lines:
        m = _RE_PAGE_LINE.match(l)
        if m:
            page = int(m.group(1))
        page_at.append(page)

    units = outline(lines)
    tok = lambda a, b: count_tokens(strip_markers("\n".join(lines[a:b])))

    # group units into chapters, then decide the spans (start, end, chapter unit, section unit | None)
    chapters: list[list[Unit]] = []
    for u in units:
        if u.level == 1:
            chapters.append([u])
        else:
            chapters[-1].append(u)

    spans: list[tuple[int, int, Unit, Unit | None, int, int]] = []  # + part, parts
    for chapter in chapters:
        head = chapter[0]
        c_start, c_end = chapter[0].start, chapter[-1].end
        if tok(c_start, c_end) <= max_tokens or len(chapter) == 1:
            cands = [(c_start, c_end, None)]
        else:
            cands = [(u.start, u.end, u if u.level == 2 else None) for u in chapter]
            # merge tiny candidates into the following sibling (or the previous one if last)
            merged: list[tuple[int, int, Unit | None]] = []
            carry: tuple[int, Unit | None] | None = None
            for a, b, sec in cands:
                if carry:
                    a, sec = carry[0], _join_sections(carry[1], sec)
                    carry = None
                if tok(a, b) < min_tokens:
                    carry = (a, sec)
                    continue
                merged.append((a, b, sec))
            if carry:
                if merged:
                    pa, _, psec = merged[-1]
                    merged[-1] = (pa, c_end, _join_sections(psec, carry[1]))
                else:
                    merged.append((carry[0], c_end, carry[1]))
            cands = merged
        for a, b, sec in cands:
            pieces = _split_span(lines, a, b, max_tokens) if tok(a, b) > max_tokens else [(a, b)]
            for k, (pa, pb) in enumerate(pieces, 1):
                spans.append((pa, pb, head, sec, k if len(pieces) > 1 else 0, len(pieces)))

    doc_code = meta.get("doc_code") or (meta.get("source") or "doc").split("_")[0]
    doc_title = meta.get("title") or meta.get("source") or doc_code
    parents: list[Parent] = []
    for a, b, head, sec, part, parts in spans:
        raw = "\n".join(lines[a:b])
        text = strip_markers(raw)
        if not text:
            continue
        content = [i for i in range(a, b) if lines[i].strip() and not _RE_PAGE_LINE.match(lines[i])]
        chapter_label = f"{head.number} {head.title}".strip() if head.number != "0" else head.title
        crumb = [doc_title, chapter_label]
        if sec:
            crumb.append(f"{sec.number} {sec.title}")
        if part:
            crumb[-1] += f" (parte {part}/{parts})"
        parents.append(
            Parent(
                parent_id=f"{doc_code}:{len(parents):03d}",
                doc_code=doc_code,
                source=meta.get("source") or "",
                doc_title=doc_title,
                language=meta.get("language") or "",
                chapter_no=head.number,
                chapter_title=head.title,
                section_no=sec.number if sec else "",
                section_title=sec.title if sec else "",
                part=part,
                parts=parts,
                title=" > ".join(crumb),
                page_start=page_at[content[0]] if content else page_at[a],
                page_end=page_at[content[-1]] if content else page_at[b - 1],
                char_start=offsets[a],
                char_end=offsets[b] - 1,
                tokens=count_tokens(text),
                text=text,
                raw=raw,
            )
        )
    return parents
