import re

import pandas as pd

from negconj.segment import is_match, merge_reason, merge_splits, segment_units, split_spaced


def toks(text, start=0):
    """Whitespace tokens with char offsets, splitting a final '.' off words."""
    out = []
    for m in re.finditer(r"\S+", text):
        w, s = m.group(0), m.start() + start
        if len(w) > 1 and w.endswith(".") and not re.fullmatch(r"(?:[A-Z]\.)+|No\.|Pub\.|L\.|Stat\.", w):
            out += [(w[:-1], s, s + len(w) - 1), (".", s + len(w) - 1, s + len(w))]
        else:
            out.append((w, s, s + len(w)))
    return out


def split_tokenizer(texts):
    """Fake tokenizer: splits after every '.', ':', '—' token (over-splits on purpose)."""
    res = []
    for t in texts:
        sents, cur = [], []
        for tk in toks(t):
            cur.append(tk)
            if tk[0] in (".", ":") or tk[0].endswith(("—", ".")):
                sents.append(cur)
                cur = []
        if cur:
            sents.append(cur)
        res.append(sents)
    return res


def test_regex():
    assert is_match("The defendant does NOT have a and b")
    assert not is_match("A and B shall not apply")          # and precedes not
    assert not is_match("cannot sell and buy")              # whole words only
    assert not is_match("not sell andirons")
    assert is_match("shall not\nact and")


def test_merge_reasons():
    assert merge_reason(toks("this chapter—"), toks("A case shall")) == "nonfinal"
    assert merge_reason(toks("Provided,"), toks("That x")) == "nonfinal"
    assert merge_reason(toks("88 Stat. 939."), toks("any employee")) == "lower"
    assert merge_reason(toks("see Pub."), toks("L. 93-259")) == "abbrev"
    assert merge_reason(toks("under 29 U.S.C."), toks("201 applies.")) == "abbrev"
    assert merge_reason(toks("this title."), toks("The term")) is None
    assert merge_reason(toks("(42 U.S.C. 1 et seq.)."), toks("The term")) is None
    assert merge_reason(toks("et seq."), toks("The term")) is None
    assert merge_reason(toks("of the section)."), toks("A person")) is None


def test_merge_splits_counts():
    sents = [toks("The defendant does not have—"), toks("A x;"), toks("B y; and"),
             toks("C z."), toks("The end.")]
    out, merged, reasons = merge_splits(sents)
    assert len(out) == 2 and merged == [3, 0]
    assert reasons["nonfinal"] == 3


def test_split_spaced_tokens():
    # Stanza's English tokenizer sometimes emits "United States" as one token.
    text = "the United States and 180 days"
    toks_in = [("the", 0, 3), ("United States", 4, 17), ("and", 18, 21), ("180 days", 22, 30)]
    out, n = split_spaced(toks_in, text)
    assert n == 2
    assert out == [("the", 0, 3), ("United", 4, 10), ("States", 11, 17), ("and", 18, 21),
                   ("180", 22, 25), ("days", 26, 30)]
    assert all(text[s:e] == w for w, s, e in out)


def test_segment_units_ids_text_and_match():
    units = pd.DataFrame({
        "unit_id": ["u1", "u2"], "section_key": ["/s1", "/s2"], "title": ["99", "99"],
        "text_parse": ["The defendant does not have— a prior offense; and a second one. "
                       "Information disclosed may be used.",
                       "Nothing here and there. It is not over."],
    })
    sents, stats = segment_units(units, split_tokenizer)
    assert sents["sent_id"].tolist() == ["u1_0", "u1_1", "u2_0", "u2_1"]
    assert sents.loc[0, "text"] == "The defendant does not have— a prior offense; and a second one."
    assert sents["matched"].tolist() == [True, False, False, False]
    assert sents.loc[0, "n_merged"] == 1 and stats["merges"] == 1
    assert stats["raw_sentences"] == 5 and stats["sentences"] == 4
    assert sents["section"].tolist() == ["/s1", "/s1", "/s2", "/s2"]
