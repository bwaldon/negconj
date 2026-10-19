"""uslm_extract.py — extract primary statutory text from USLM XML (U.S. Code).

Ports the procedure in ``Another_copy_of_extract.ipynb`` into a reusable module:
notes are stripped, primary text is taken from <content>, <chapeau>,
<continuation> and <proviso> elements, every element gets a
(parent_id, parent_position) locator, and chapeau constructions are condensed
into single sentences.

Fixes relative to the notebook
------------------------------
1. Text joins: text is rendered with a space at every *block* boundary (p,
   content, chapeau, level elements, num, ...) and none at inline tags (ref, i,
   b, date, ...). The notebook's bare ``get_text()`` glued adjacent <p> blocks
   ("prosperity.The"); ``get_text(" ")`` would instead split words around
   inline tags. Unicode spaces (U+00A0, U+202F, ...) are normalized and runs
   of whitespace collapsed.
2. Footnote markers: <ref class="footnoteRef"> is removed along with
   <note>/<notes>, so markers like the "1" in "section 134(d); 1" are gone.
   So is a bare digit-only <sup> (a footnote cited a second time), unless it
   follows a digit or minus sign (an exponent: "10−<sup>4</sup>").
3. Positions: ``parent_position`` is the element's index among the *Tag*
   descendants of its @id-bearing anchor (the anchor itself is 0), found by
   object identity. The notebook used ``list(parent.children).index(el)``,
   which (a) raises when the element is not a direct child of the anchor,
   (b) matches by structural equality, so identical siblings collide, and
   (c) crashed when the element itself carried the @id. Note that positions
   therefore differ numerically from the notebook's (whitespace strings are
   no longer counted).
4. @identifier: each row also records the nearest @identifier (e.g.
   ``/us/usc/t29/s2/a/1``), a human-readable citation path that is stable
   across release points, unlike the opaque @id, and the @identifier of its
   enclosing <section> (``section``), the checkpoint key for downstream
   per-section processing.
5. Chapeau condensation follows document structure, not indent classes: a
   chapeau-bearing level is rendered by walking its *direct* children in
   order (chapeau, sub-levels recursively, continuation(s), provisos). The
   notebook's recursive ``find_all(class=indent{n+1})`` /
   ``find('continuation')`` could pick up items or a continuation belonging to
   a nested structure, crashed when no class attribute was present, and
   joined the continuation with no space.
6. Two strings per unit: ``text_display`` keeps sub-level numbers ("(A)",
   "(ii)"); ``text_parse`` drops them so the parser doesn't see -LRB- A -RRB-.
   Sub-level headings are dropped from both.
7. No double counting: only the *outermost* chapeau-bearing level in a
   subtree becomes a condensed unit; every text element inside it is marked
   ``consumed_by`` that unit and is not emitted again as a standalone unit.
   Text inside <quotedContent> is rendered as part of its enclosing element
   and never extracted or condensed on its own.
8. Rows are collected in lists and turned into DataFrames once (the
   notebook's per-row ``pd.concat`` is quadratic).

Usage
-----
    from uslm_extract import extract_title
    elements, units = extract_title("usc29.xml")

    python -I uslm_extract.py usc29.xml usc18.xml --out-dir extracted/

``units`` is the table for the analysis (one row per standalone element or
condensed chapeau sentence); ``elements`` is the element-level table that
mirrors the notebook's output, for traceability.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup, Tag
from bs4.element import (
    Comment,
    Declaration,
    Doctype,
    NavigableString,
    ProcessingInstruction,
)

__all__ = [
    "TEXT_ELEMENTS",
    "LEVEL_ELEMENTS",
    "load_title",
    "strip_non_primary",
    "normalize_ws",
    "render_text",
    "assemble_level",
    "get_id",
    "get_identifier",
    "get_section",
    "PositionIndex",
    "extract_title",
]

# Elements that hold primary statutory text (notebook cell 7).
TEXT_ELEMENTS = ("content", "chapeau", "continuation", "proviso")

# USLM hierarchical elements a chapeau can introduce.
LEVEL_ELEMENTS = frozenset({
    "title", "subtitle", "chapter", "subchapter", "part", "subpart",
    "division", "subdivision", "article", "subarticle",
    "section", "subsection", "paragraph", "subparagraph",
    "clause", "subclause", "item", "subitem", "subsubitem", "level",
})

# Elements whose boundaries are word boundaries. Everything else (ref, i, b,
# date, term, inline, span, sup, shortTitle, quotedText, ...) is inline.
BLOCK_ELEMENTS = frozenset(TEXT_ELEMENTS) | LEVEL_ELEMENTS | frozenset({
    "p", "num", "heading", "subheading", "quotedContent", "block", "list",
    "listItem", "table", "thead", "tbody", "tr", "td", "th", "br",
    "layout", "row", "column",
})

_SKIP_STRINGS = (Comment, Declaration, Doctype, ProcessingInstruction)
_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200b\u202f\u205f\u3000]")
_WS = re.compile(r"\s+")
_FOOTNOTE_NUM = re.compile(r"\d{1,2}")
_EXPONENT_BASE = re.compile(r"[0-9−\-]$")


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------

def load_title(path: str | Path) -> BeautifulSoup:
    """Parse one USLM title file with the XML parser (as in the notebook)."""
    with open(path, "rb") as fh:
        return BeautifulSoup(fh.read(), "xml")


def _classes(tag: Tag) -> list[str]:
    cls = tag.get("class")
    if cls is None:
        return []
    return cls.split() if isinstance(cls, str) else list(cls)


def strip_non_primary(soup: BeautifulSoup) -> dict[str, int]:
    """Remove notes and footnote markers in place; return removal counts.

    <note> elements never carry primary statutory text (notebook cell 4).
    <notes> wrappers go too. <ref class="footnoteRef"> is the inline marker
    pointing at a footnote note; once the note is gone the marker is noise.
    """
    counts = {"note": 0, "notes": 0, "footnoteRef": 0, "footnoteSup": 0}
    for name in ("notes", "note"):
        for tag in soup.find_all(name):
            if tag.decomposed:  # already removed with an enclosing <notes>
                continue
            tag.decompose()
            counts[name] += 1
    for ref in soup.find_all("ref"):
        if "footnoteRef" in _classes(ref):
            ref.decompose()
            counts["footnoteRef"] += 1
    # A footnote cited a second time is marked with a bare <sup>1</sup> (no
    # <ref class="footnoteRef">), e.g. 29 U.S.C. 218c(a)(2) at 119-121.
    # Exponents ("10−<sup>4</sup> Pa") follow a digit or minus sign; keep those.
    for sup in soup.find_all("sup"):
        if not _FOOTNOTE_NUM.fullmatch(sup.get_text().strip()):
            continue
        prev = sup.previous_sibling
        if isinstance(prev, NavigableString) and _EXPONENT_BASE.search(str(prev)):
            continue
        sup.decompose()
        counts["footnoteSup"] += 1
    return counts


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------

def normalize_ws(text: str) -> str:
    return _WS.sub(" ", _UNICODE_SPACES.sub(" ", text)).strip()


def render_text(el: Tag, *, keep_num: bool = True) -> str:
    """Render an element's text with spaces only at block boundaries."""
    parts: list[str] = []

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if not isinstance(child, _SKIP_STRINGS):
                    parts.append(str(child))
            elif isinstance(child, Tag):
                if child.name == "num" and not keep_num:
                    parts.append(" ")
                    continue
                block = child.name in BLOCK_ELEMENTS
                if block:
                    parts.append(" ")
                walk(child)
                if block:
                    parts.append(" ")

    walk(el)
    return normalize_ws("".join(parts))


def assemble_level(level: Tag, *, keep_num: bool = True) -> str:
    """Condense a chapeau-bearing level into one string.

    Walks the level's direct children in document order: text elements
    (chapeau, content, continuation, proviso) are rendered; sub-levels are
    assembled recursively, prefixed with their <num> when ``keep_num``.
    The level's own <num> and all <heading>s are omitted.
    """
    pieces: list[str] = []
    for child in level.children:
        if not isinstance(child, Tag):
            continue
        if child.name in TEXT_ELEMENTS:
            pieces.append(render_text(child, keep_num=keep_num))
        elif child.name in LEVEL_ELEMENTS:
            body = assemble_level(child, keep_num=keep_num)
            if keep_num:
                num = child.find("num", recursive=False)
                if num is not None:
                    body = render_text(num) + " " + body
            pieces.append(body)
    return normalize_ws(" ".join(pieces))


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------

def _anchor(el: Tag) -> Tag | None:
    """The element itself if it has @id, else its nearest @id ancestor."""
    if el.get("id"):
        return el
    return el.find_parent(attrs={"id": True})


def get_id(el: Tag) -> str | None:
    anchor = _anchor(el)
    return anchor.get("id") if anchor is not None else None


def get_identifier(el: Tag) -> str | None:
    if el.get("identifier"):
        return el.get("identifier")
    holder = el.find_parent(attrs={"identifier": True})
    return holder.get("identifier") if holder is not None else None


def get_section(el: Tag) -> str | None:
    """@identifier of the enclosing <section> (or the element itself if it is
    one). Used as the checkpoint key downstream. None for text that sits
    outside any section."""
    sec = el if el.name == "section" else el.find_parent("section")
    if sec is None:
        return None
    return sec.get("identifier") or sec.get("id")


class PositionIndex:
    """Position of an element among the Tag descendants of its @id anchor.

    The anchor itself is position 0; its Tag descendants are numbered from 1
    in document order. Lookup is by object identity, so identical siblings
    get distinct positions. One index is built per anchor and cached.
    Build it only after the tree is final (after ``strip_non_primary``).
    """

    def __init__(self) -> None:
        self._cache: dict[int, dict[int, int]] = {}

    def position(self, el: Tag) -> tuple[str | None, int | None]:
        anchor = _anchor(el)
        if anchor is None:
            return None, None
        index = self._cache.get(id(anchor))
        if index is None:
            index = {id(anchor): 0}
            i = 0
            for d in anchor.descendants:
                if isinstance(d, Tag):
                    i += 1
                    index[id(d)] = i
            self._cache[id(anchor)] = index
        return anchor.get("id"), index[id(el)]


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _in_quoted(el: Tag) -> bool:
    return el.find_parent("quotedContent") is not None


def _title_from_path(path: Path) -> tuple[str, bool]:
    m = re.search(r"usc(\d+a?)", path.stem, flags=re.IGNORECASE)
    label = m.group(1).lower() if m else path.stem
    return label, label.endswith("a")


def extract_title(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract one USLM title file.

    Returns ``(elements, units)``.

    elements — one row per <content>/<chapeau>/<continuation>/<proviso>
        outside <quotedContent>: elem_source, section, parent_id,
        parent_position, identifier, text, consumed_by.
    units — the analysis table: one row per text element not inside a
        condensed chapeau, plus one row per outermost chapeau-bearing level:
        unit_id, title, is_appendix, section, identifier, parent_id,
        parent_position,
        elem_source ('content'|'chapeau_condensed'|'continuation'|'proviso'),
        text_display, text_parse, n_elements.
    """
    path = Path(path)
    title, is_appendix = _title_from_path(path)
    soup = load_title(path)
    strip_non_primary(soup)
    positions = PositionIndex()

    # Chapeau-bearing levels, ignoring anything inside quoted (amendatory)
    # content, which is rendered as part of its enclosing element.
    chapeau_levels: dict[int, Tag] = {}
    for ch in soup.find_all("chapeau"):
        if not _in_quoted(ch) and ch.parent is not None:
            chapeau_levels[id(ch.parent)] = ch.parent

    # Outermost: no ancestor is itself a chapeau-bearing level.
    outermost: list[Tag] = []
    for lvl in chapeau_levels.values():
        if not any(id(a) in chapeau_levels for a in lvl.parents):
            outermost.append(lvl)

    units: list[dict] = []
    consumed: dict[int, str] = {}
    for lvl in outermost:
        pid, pos = positions.position(lvl)
        unit_id = f"{pid}_{pos}_chap"
        members = [d for d in lvl.descendants
                   if isinstance(d, Tag) and d.name in TEXT_ELEMENTS
                   and not _in_quoted(d)]
        for d in members:
            consumed[id(d)] = unit_id
        units.append({
            "unit_id": unit_id,
            "title": title,
            "is_appendix": is_appendix,
            "section": get_section(lvl),
            "identifier": get_identifier(lvl),
            "parent_id": pid,
            "parent_position": pos,
            "elem_source": "chapeau_condensed",
            "text_display": assemble_level(lvl, keep_num=True),
            "text_parse": assemble_level(lvl, keep_num=False),
            "n_elements": len(members),
        })

    elements: list[dict] = []
    for el in soup.find_all(list(TEXT_ELEMENTS)):
        if _in_quoted(el):
            continue
        pid, pos = positions.position(el)
        identifier = get_identifier(el)
        section = get_section(el)
        text_display = render_text(el, keep_num=True)
        owner = consumed.get(id(el))
        elements.append({
            "elem_source": el.name,
            "section": section,
            "parent_id": pid,
            "parent_position": pos,
            "identifier": identifier,
            "text": text_display,
            "consumed_by": owner,
        })
        if owner is None:
            units.append({
                "unit_id": f"{pid}_{pos}",
                "title": title,
                "is_appendix": is_appendix,
                "section": section,
                "identifier": identifier,
                "parent_id": pid,
                "parent_position": pos,
                "elem_source": el.name,
                "text_display": text_display,
                "text_parse": render_text(el, keep_num=False),
                "n_elements": 1,
            })

    elements_df = pd.DataFrame(elements, columns=[
        "elem_source", "section", "parent_id", "parent_position",
        "identifier", "text", "consumed_by"])
    units_df = pd.DataFrame(units, columns=[
        "unit_id", "title", "is_appendix", "section", "identifier", "parent_id",
        "parent_position", "elem_source", "text_display", "text_parse",
        "n_elements"])

    # Hard checks (the notebook's groupby check, made fatal).
    dup = elements_df.duplicated(["parent_id", "parent_position"])
    if dup.any():
        raise AssertionError(
            f"{path.name}: {int(dup.sum())} duplicate (parent_id, parent_position)")
    if units_df["unit_id"].duplicated().any():
        raise AssertionError(f"{path.name}: duplicate unit_id")
    if elements_df["parent_id"].isna().any():
        raise AssertionError(f"{path.name}: text element with no @id anchor")
    return elements_df, units_df


def _write(df: pd.DataFrame, stem: Path) -> Path:
    try:
        out = stem.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except ImportError:  # no pyarrow/fastparquet installed
        out = stem.with_suffix(".csv")
        df.to_csv(out, index=False)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("xml", nargs="+", type=Path, help="USLM title files")
    ap.add_argument("--out-dir", type=Path, default=Path("extracted"))
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for xml in args.xml:
        elements, units = extract_title(xml)
        title, _ = _title_from_path(xml)
        e = _write(elements, args.out_dir / f"elements_{title}")
        u = _write(units, args.out_dir / f"units_{title}")
        n_cond = int((units["elem_source"] == "chapeau_condensed").sum())
        print(f"{xml.name}: {len(elements)} elements, {len(units)} units "
              f"({n_cond} condensed chapeau) -> {e.name}, {u.name}")


if __name__ == "__main__":
    main()
