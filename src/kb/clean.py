"""Post-processing shared by all extractors: turn raw Markdown pages into clean pages.

Input/output unit is a list of page strings (index 0 = page 1). Everything here is
heuristic and deliberately generic, so it works for any manual of the family:
- repeated running headers/footers are detected by frequency across pages
- symbol-font glyphs used for warning icons are mapped to explicit tags
- hyphenation leftovers and table artefacts are repaired
"""
from __future__ import annotations

import re
from collections import Counter

# Glyphs of the proprietary symbol font (Simboli_Pz) used for warning boxes.
SYMBOL_TAGS = {"d": "[!DANGER]", "a": "[!WARNING]", "i": "[!INFO]"}

_HEADER_MIN_SHARE = 0.4  # a short line present on >= 40% of pages is a running header/footer
_HEADER_MAX_LEN = 120
_CAPS_MIN_PAGES = 5

_RE_MARK = re.compile(r"</?mark>")
_RE_PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)
_RE_PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
_RE_MULTI_NL = re.compile(r"\n{3,}")
# pymupdf4llm renders the icon as an inline code span: "- `d` **text**"
_RE_SYMBOL_CODE = re.compile(r"(^|\s)`([dai])`\s*(?=\S)", re.M)
# docling glues the glyph to the first word: "aNel locale", "dÈ vietato", "- iInformazioni"
_RE_SYMBOL_GLUED = re.compile(r"^((?:- )?)([dai])(?=[A-ZÀ-ÖØ-Þ])", re.M)
# "nomi- nale" / "nomi-<br>nale" inside table cells
_RE_HYPHEN_SPACE = re.compile(r"([a-zà-öø-ÿ])-(?:\s|<br>)+(?=[a-zà-öø-ÿ])")
_RE_BR = re.compile(r"\s*<br>\s*")
_RE_SUP = re.compile(r"<sup>(.*?)</sup>")
_RE_HTML_TABLE = re.compile(r"<table>.*?</table>", re.S)
_RE_IMAGE_REF = re.compile(r"^!\[[^\]]*\]\([^)]*\)\s*$", re.M)
_RE_ESCAPED_PUNCT = re.compile(r"\\([-_*#.])")
# headings: "#### **4.2 PRESA D'ARIA**" / "## 2.8 DATI TECNICI" -> level from the section number
_RE_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)
_RE_SECTION_NO = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")


def compact_table_row(line: str) -> str:
    """'| a   | a   | 13,2 |' -> '| a | 13,2 |': strip padding, drop cells repeated by merged-cell expansion."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if all(set(c) <= set("-: ") for c in cells):  # separator row
        return "|" + "|".join("---" for _ in cells) + "|"
    out: list[str] = []
    for c in cells:
        if out and c == out[-1] and re.search(r"[^\W\d_]{3}", c):  # repeated text, never numbers
            continue
        out.append(re.sub(r"\s{2,}", " ", c))
    return "| " + " | ".join(out) + " |"


def _norm(line: str) -> str:
    return re.sub(r"\W+", " ", line).strip().lower()


def detect_running_lines(pages: list[str]) -> set[str]:
    """Return normalized lines that repeat on many pages (headers, footers, brand)."""
    counts: Counter[str] = Counter()
    for page in pages:
        seen = {_norm(l) for l in page.splitlines() if 0 < len(l.strip()) <= _HEADER_MAX_LEN}
        counts.update(seen)
    threshold = max(3, int(len(pages) * _HEADER_MIN_SHARE))
    running = {l for l, c in counts.items() if c >= threshold and l and not l.isdigit()}
    # an ALL-CAPS line (outside tables) repeated on many pages is a running title that the
    # layout model sometimes classifies as a section heading instead of page furniture
    caps: Counter[str] = Counter()
    for page in pages:
        seen = {
            _norm(l) for l in page.splitlines()
            if not l.lstrip().startswith("|") and len(l.strip()) >= 10 and l.strip("#* ").isupper()
        }
        caps.update(seen)
    running |= {l for l, c in caps.items() if c >= _CAPS_MIN_PAGES}
    return running


def html_table_to_markdown(html: str) -> str:
    """Minimal <table> -> Markdown pipe table (colspan padded with empty cells, rowspan ignored)."""
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self.cell: list[str] | None = None
            self.span = 1

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.rows.append([])
            elif tag in ("td", "th"):
                self.cell = []
                self.span = int(dict(attrs).get("colspan", 1) or 1)

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.cell is not None and self.rows:
                text = re.sub(r"\s+", " ", "".join(self.cell)).strip()
                self.rows[-1].append(text)
                self.rows[-1].extend([""] * (self.span - 1))
                self.cell = None

        def handle_data(self, data):
            if self.cell is not None:
                self.cell.append(data)

    p = _P()
    p.feed(html)
    rows = [r for r in p.rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join("---" for _ in range(width)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _normalize_heading(m: re.Match) -> str:
    title = m.group(1).strip().strip("*_ ").replace("**", "").replace("<u>", "").replace("</u>", "")
    sec = _RE_SECTION_NO.match(title)
    # "1 AVVERTENZE GENERALI" is a chapter; "1 Key ON/OFF" is a numbered list item styled bold
    if sec and ("." in sec.group(1) or sec.group(2).isupper()):
        level = min(sec.group(1).count(".") + 1, 3)
        return f"{'#' * level} {sec.group(1)} {sec.group(2).strip()}"
    return f"#### {title}"


def prepass(text: str, drop_picture_text: bool = True) -> str:
    """Text-level rewrites applied before running-line detection."""
    if drop_picture_text:
        text = _RE_PICTURE_TEXT.sub("", text)
    text = _RE_MARK.sub("", text)
    text = _RE_SYMBOL_CODE.sub(lambda m: f"{m.group(1)}**{SYMBOL_TAGS[m.group(2)]}** ", text)
    text = _RE_SYMBOL_GLUED.sub(lambda m: f"{m.group(1)}**{SYMBOL_TAGS[m.group(2)]}** ", text)
    text = _RE_SUP.sub(r"\1", text)
    text = _RE_HTML_TABLE.sub(lambda m: "\n" + html_table_to_markdown(m.group(0)) + "\n", text)
    text = _RE_IMAGE_REF.sub("", text)
    text = _RE_ESCAPED_PUNCT.sub(r"\1", text)
    text = _RE_HEADING.sub(_normalize_heading, text)
    return text


def clean_page(text: str, running: set[str]) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _norm(stripped) in running:
            continue
        if _RE_PAGE_NUMBER.match(stripped):
            continue
        if stripped.startswith("|"):
            line = compact_table_row(_RE_HYPHEN_SPACE.sub(r"\1", _RE_BR.sub(" ", line)))
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = _RE_HYPHEN_SPACE.sub(r"\1", text)
    text = _RE_MULTI_NL.sub("\n\n", text)
    return text.strip() + "\n"


def clean_pages(pages: list[str], drop_picture_text: bool = True) -> list[str]:
    pages = [prepass(p, drop_picture_text) for p in pages]
    running = detect_running_lines(pages)
    return [clean_page(p, running) for p in pages]
