"""Pulsifer regression (work plan §3.4, §7.1): 18 U.S.C. § 3553(f) must come out
as one condensed unit holding the chapeau's negation and the `and` before (C),
and must survive every stage through to the classified table.

Needs data/raw/usc18.xml. Later stages are checked against the results directory
in $NEGCONJ_RESULTS (default data/local/results) when it exists.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / "data" / "raw" / "usc18.xml"
RESULTS = Path(os.environ.get("NEGCONJ_RESULTS", ROOT / "data" / "local" / "results"))
PULSIFER = "/us/usc/t18/s3553/f"
NEG = "the defendant does not have— more than 4 criminal history points"
AND_C = "; and a prior 2-point violent offense"

needs_xml = pytest.mark.skipif(not XML.exists(), reason="usc18.xml not downloaded")


@pytest.fixture(scope="module")
def pulsifer_unit():
    from negconj.uslm_extract import extract_title
    _, units = extract_title(XML)
    hit = units[units["identifier"].fillna("").str.startswith(PULSIFER)]
    return hit


@needs_xml
def test_one_condensed_unit(pulsifer_unit):
    assert len(pulsifer_unit) == 1
    row = pulsifer_unit.iloc[0]
    assert row["elem_source"] == "chapeau_condensed" and row["identifier"] == PULSIFER
    assert NEG in row["text_parse"] and AND_C in row["text_parse"]
    assert row["text_parse"].index(NEG) < row["text_parse"].index(AND_C)
    assert "(C) a prior 2-point violent offense" in row["text_display"]


@needs_xml
def test_segmented_into_one_matching_sentence(pulsifer_unit):
    pytest.importorskip("stanza")
    from negconj.segment import build_tokenizer, segment_units, stanza_tokenizer
    from negconj.shards import prefilter
    pre = prefilter(pulsifer_unit)
    assert len(pre) == 1
    sents, _ = segment_units(pre, stanza_tokenizer(build_tokenizer()))
    s = sents[sents["text"].str.contains(NEG, regex=False)]
    assert len(s) == 1 and AND_C in s.iloc[0]["text"] and s.iloc[0]["matched"]


@pytest.mark.skipif(not (RESULTS / "sentences_classified.parquet").exists(),
                    reason="no classified results yet")
def test_survives_to_classified_table():
    parses = pd.read_parquet(RESULTS / "parses.parquet")
    classified = pd.read_parquet(RESULTS / "sentences_classified.parquet")
    for df in (parses, classified):
        if "identifier" not in df:
            df = df.merge(classified[["sent_id", "identifier"]], on="sent_id")
        hit = df[(df["identifier"] == PULSIFER) & df["text"].str.contains(NEG, regex=False)]
        assert len(hit) == 1
        assert AND_C in hit.iloc[0]["text"] and hit.iloc[0]["tree"].startswith("(ROOT")
    row = classified[(classified["identifier"] == PULSIFER)
                     & classified["text"].str.contains(NEG, regex=False)].iloc[0]
    assert row["flagged"] in (True, False) and row["n_pairs"] >= 1
