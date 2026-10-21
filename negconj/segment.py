"""Sentence segmentation of units and the `not … and` / `not … or` regex (work plan §5.4).

Stanza's tokenizer does the splitting; a post-pass re-merges splits that are
not sentence boundaries in statutory text. A split is undone when:

  nonfinal  the left sentence does not end in . ? ! (ignoring closing
            brackets/quotes): splits after "—", ",", ":", ";", ")" — the
            condensed-chapeau join point ("this chapter— A case shall…")
            and provisos (": Provided, That…").
  lower     the right sentence starts with a lowercase letter.
  punct     the right sentence starts with closing punctuation.
  abbrev    the left sentence ends in an abbreviation that never ends a
            sentence ("No.", "Pub.", "L.", "U.S.C.", "Jan.", …).
            "et seq." can end a sentence, so it only merges via `lower`.

Each sentence gets ID ``{unit_id}_{chunkno}`` (chunkno from 0).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Sequence

import pandas as pd

# Connective -> sentence regex. "and" is the negative-conjunction study; "or" mirrors it
# for negative disjunction (same shape: not precedes the connective, whole words).
CONNECTIVES = ("and", "or")
MATCH_RES = {c: re.compile(rf"\bnot\b.*\b{c}\b", re.IGNORECASE | re.DOTALL) for c in CONNECTIVES}
MATCH_RE = MATCH_RES["and"]

# Tokens (as Stanza emits them, or with a split-off final ".") that never end
# a sentence in the Code.
ABBREV_NONFINAL = frozenset({
    "No.", "Nos.", "Pub.", "L.", "U.S.C.", "Stat.", "v.", "vs.",
    "Sec.", "Secs.", "sec.", "secs.", "subsec.", "subsecs.", "par.", "pars.",
    "subpar.", "subpars.", "cl.", "subcl.", "ch.", "chs.", "Ch.", "subch.",
    "pt.", "Pt.", "art.", "Art.", "Mr.", "Mrs.", "Ms.", "Dr.", "Gen.",
    "Rept.", "Doc.", "Res.", "Con.", "H.R.", "Exec.", "Ord.", "Reorg.",
    "Fed.", "Reg.", "C.F.R.", "e.g.", "i.e.", "cf.", "Cf.", "approx.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.", "Sep.", "Oct.", "Nov.", "Dec.",
})
_FINAL = (".", "?", "!")
_CLOSERS = frozenset({")", "]", "”", "’", "\"", "'", "''"})

# A tokenizer maps texts to, per text, a list of sentences, each a list of
# (token_text, start_char, end_char).
Tok = tuple[str, int, int]
Tokenizer = Callable[[Sequence[str]], list[list[list[Tok]]]]

SENT_COLUMNS = ["sent_id", "unit_id", "section", "title", "text", "tokens",
                "n_tokens", "n_merged", "matched"]


def is_match(text: str, conn: str = "and") -> bool:
    return MATCH_RES[conn].search(text) is not None


def _last_word(tokens: list[Tok]) -> tuple[str, bool]:
    """Last non-closer token text, and whether it is/ends with a final mark."""
    i = len(tokens) - 1
    while i > 0 and tokens[i][0] in _CLOSERS:
        i -= 1
    last = tokens[i][0]
    if last == "." and i > 0:
        return tokens[i - 1][0] + ".", True
    return last, last.endswith(_FINAL)


def merge_reason(left: list[Tok], right: list[Tok]) -> str | None:
    word, final = _last_word(left)
    if not final:
        return "nonfinal"
    first = right[0][0]
    if first[:1].islower():
        return "lower"
    if first in _CLOSERS or first[:1] in ",;:)]":
        return "punct"
    if word in ABBREV_NONFINAL:
        return "abbrev"
    return None


def merge_splits(sents: list[list[Tok]]) -> tuple[list[list[Tok]], list[int], Counter]:
    """Re-merge spurious splits. Returns (sentences, merges per sentence, reasons)."""
    out: list[list[Tok]] = []
    merged: list[int] = []
    reasons: Counter = Counter()
    for s in sents:
        if out:
            r = merge_reason(out[-1], s)
            if r is not None:
                out[-1] = out[-1] + s
                merged[-1] += 1
                reasons[r] += 1
                continue
        out.append(list(s))
        merged.append(0)
    return out, merged, reasons


def segment_units(units: pd.DataFrame, tokenizer: Tokenizer,
                  conn: str = "and") -> tuple[pd.DataFrame, dict]:
    """Segment ``text_parse`` of each unit; flag matches of ``MATCH_RES[conn]``.

    ``units`` needs unit_id, section_key, title, text_parse. Returns the
    sentence table (SENT_COLUMNS; ``section`` holds the section key) and
    stats {raw_sentences, sentences, merges, merge_<reason>...}.
    """
    texts = list(units["text_parse"])
    tokenized = tokenizer(texts) if texts else []
    rows: list[dict] = []
    stats: Counter = Counter()
    for (uid, sec, title, text), sents in zip(
            units[["unit_id", "section_key", "title", "text_parse"]].itertuples(index=False),
            tokenized):
        stats["raw_sentences"] += len(sents)
        sents, n_merged, reasons = merge_splits(sents)
        stats.update({f"merge_{k}": v for k, v in reasons.items()})
        stats["merges"] += sum(n_merged)
        for k, (s, m) in enumerate(zip(sents, n_merged)):
            stext = text[s[0][1]:s[-1][2]]
            toks = [t[0] for t in s]
            rows.append({
                "sent_id": f"{uid}_{k}", "unit_id": uid, "section": sec,
                "title": title, "text": stext, "tokens": toks,
                "n_tokens": len(toks), "n_merged": m, "matched": is_match(stext, conn),
            })
    stats["sentences"] = len(rows)
    return pd.DataFrame(rows, columns=SENT_COLUMNS), dict(stats)


_NONSPACE = re.compile(r"\S+")


def split_spaced(tokens: list[Tok], text: str) -> tuple[list[Tok], int]:
    """Split tokens that contain whitespace ("United States", "180 days" —
    Stanza's English tokenizer occasionally emits these) at the whitespace.
    A leaf with a space can't round-trip through a bracketed tree. Returns
    (tokens, number of tokens split)."""
    out: list[Tok] = []
    n = 0
    for word, start, end in tokens:
        if not any(c.isspace() for c in word):
            out.append((word, start, end))
            continue
        n += 1
        out += [(m.group(0), start + m.start(), start + m.end())
                for m in _NONSPACE.finditer(text[start:end])]
    return out, n


def stanza_tokenizer(nlp) -> Tokenizer:
    """Wrap a Stanza tokenize pipeline as a Tokenizer. Spaced tokens are
    split and counted in ``tok.stats['spaced_tokens_split']``."""
    stats: Counter = Counter()

    def tok(texts: Sequence[str]) -> list[list[list[Tok]]]:
        docs = nlp.bulk_process(list(texts))
        res = []
        for text, d in zip(texts, docs):
            sents = []
            for s in d.sentences:
                toks, n = split_spaced([(t.text, t.start_char, t.end_char) for t in s.tokens], text)
                stats["spaced_tokens_split"] += n
                sents.append(toks)
            res.append(sents)
        return res
    tok.stats = stats
    return tok


def build_tokenizer(use_gpu: bool = False):
    import stanza
    return stanza.Pipeline("en", processors="tokenize", use_gpu=use_gpu,
                           logging_level="WARN", download_method=None)
