"""Tests for uslm_extract.py. Runs under pytest, or directly: python -I test_uslm_extract.py

Uses a small synthetic USLM file that reproduces the structures the notebook's
fixes target. Once real titles are downloaded, run extract_title on usc29.xml
and usc18.xml as well (see check_real_titles at the bottom).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "negconj"))
from uslm_extract import extract_title  # noqa: E402

FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<uscDoc xmlns="http://xml.house.gov/schemas/uslm/1.0" identifier="/us/usc/t99">
<main><title id="t99" identifier="/us/usc/t99">
<section id="s1" identifier="/us/usc/t99/s1"><num value="1">§\u202f1.</num><heading> Plain section</heading>
<content>
<p class="indent0">The first paragraph ends here.</p><p class="indent0">The second paragraph follows.</p>
</content>
<content id="own">An element that carries its own id.</content>
<notes><note><heading>Editorial Notes</heading><p>This note text must not appear.</p></note></notes>
</section>
<section id="s2" identifier="/us/usc/t99/s2"><num value="2">§\u202f2.</num><heading> Safety valve</heading>
<subsection id="s2a" identifier="/us/usc/t99/s2/a" class="indent0"><num value="a">(a)</num><heading> In general</heading>
<chapeau class="indent0">The defendant does not have—</chapeau>
<paragraph id="s2a1" identifier="/us/usc/t99/s2/a/1" class="indent1"><num value="1">(1)</num>
<content>more than 4 criminal history points<ref class="footnoteRef" idref="fn1">1</ref>;</content></paragraph>
<paragraph id="s2a2" identifier="/us/usc/t99/s2/a/2" class="indent1"><num value="2">(2)</num>
<chapeau class="indent1">a prior offense that—</chapeau>
<subparagraph id="s2a2A" identifier="/us/usc/t99/s2/a/2/A" class="indent2"><num value="A">(A)</num><content>is serious; or</content></subparagraph>
<subparagraph id="s2a2B" identifier="/us/usc/t99/s2/a/2/B" class="indent2"><num value="B">(B)</num><content>is violent;</content></subparagraph>
<continuation class="indent1">as determined by the court; and</continuation>
</paragraph>
<paragraph id="s2a3" identifier="/us/usc/t99/s2/a/3" class="indent1"><num value="3">(3)</num>
<content>a prior <i>violent</i> offense under section <ref href="/us/usc/t18/s16">16</ref>,</content></paragraph>
<continuation class="indent0">as determined under the sentencing guidelines.</continuation>
<notes><note type="footnote" id="fn1"><p>1 So in original.</p></note></notes>
</subsection>
<subsection id="s2b" identifier="/us/usc/t99/s2/b"><num value="b">(b)</num>
<content>Identical text.</content><content>Identical text.</content>
</subsection>
<subsection id="s2c" identifier="/us/usc/t99/s2/c"><num value="c">(c)</num>
<content>Section 5 is amended by adding at the end <quotedContent><subsection><num>(z)</num><chapeau>A quoted chapeau—</chapeau><paragraph><num>(1)</num><content>quoted item</content></paragraph></subsection></quotedContent>.</content>
</subsection>
</section>
</title></main></uscDoc>
"""


def _extract():
    d = Path(tempfile.mkdtemp())
    p = d / "usc99.xml"
    p.write_text(FIXTURE, encoding="utf-8")
    return extract_title(p)


def test_notes_and_footnote_refs_removed():
    elements, units = _extract()
    blob = " ".join(units["text_display"]) + " ".join(elements["text"])
    assert "note text must not appear" not in blob
    assert "So in original" not in blob
    assert "points;" in blob and "points 1" not in blob


def test_block_boundaries_spaced_inline_not_split():
    _, units = _extract()
    s1 = units[units["identifier"] == "/us/usc/t99/s1"]["text_display"].tolist()
    assert "The first paragraph ends here. The second paragraph follows." in s1
    cond = units[units["elem_source"] == "chapeau_condensed"].iloc[0]
    assert "a prior violent offense under section 16," in cond["text_parse"]


def test_unicode_spaces_normalized():
    _, units = _extract()
    assert not any("\u202f" in t or "\u00a0" in t for t in units["text_display"])


def test_positions_unique_identity_and_self_id():
    elements, _ = _extract()
    b = elements[elements["identifier"] == "/us/usc/t99/s2/b"]
    assert len(b) == 2 and b["parent_position"].nunique() == 2
    own = elements[elements["parent_id"] == "own"]
    assert own["parent_position"].tolist() == [0]
    assert not elements.duplicated(["parent_id", "parent_position"]).any()


def test_chapeau_condensed_once_structurally():
    elements, units = _extract()
    cond = units[units["elem_source"] == "chapeau_condensed"]
    assert len(cond) == 1, "only the outermost chapeau level becomes a unit"
    row = cond.iloc[0]
    assert row["identifier"] == "/us/usc/t99/s2/a"
    assert row["text_parse"] == (
        "The defendant does not have— more than 4 criminal history points; "
        "a prior offense that— is serious; or is violent; as determined by "
        "the court; and a prior violent offense under section 16, as "
        "determined under the sentencing guidelines.")
    assert row["text_display"].startswith(
        "The defendant does not have— (1) more than 4 criminal history "
        "points; (2) a prior offense that— (A) is serious; or (B) is violent;")
    # Every text element under (a) is consumed and not re-emitted.
    under_a = elements[elements["identifier"].str.startswith("/us/usc/t99/s2/a")]
    assert under_a["consumed_by"].eq(row["unit_id"]).all()
    assert row["n_elements"] == len(under_a) == 8
    standalone = units[units["elem_source"] != "chapeau_condensed"]
    assert not standalone["identifier"].str.startswith("/us/usc/t99/s2/a").any()


def test_quoted_content_stays_inside_enclosing_element():
    elements, units = _extract()
    assert not elements["text"].eq("quoted item").any()
    c = units[units["identifier"] == "/us/usc/t99/s2/c"]
    assert len(c) == 1 and c.iloc[0]["elem_source"] == "content"
    assert "A quoted chapeau— (1) quoted item" in c.iloc[0]["text_display"]


def test_section_column():
    elements, units = _extract()
    assert units["section"].notna().all() and elements["section"].notna().all()
    cond = units[units["elem_source"] == "chapeau_condensed"].iloc[0]
    assert cond["section"] == "/us/usc/t99/s2"
    assert set(units["section"]) == {"/us/usc/t99/s1", "/us/usc/t99/s2"}
    # every unit's identifier lies within its section
    assert all(i.startswith(s) for i, s in zip(units["identifier"], units["section"]))


SUP_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<uscDoc xmlns="http://xml.house.gov/schemas/uslm/1.0" identifier="/us/usc/t98">
<main><title id="t98" identifier="/us/usc/t98">
<section id="s1" identifier="/us/usc/t98/s1"><num value="1">§ 1.</num>
<content id="c1">any provision of this title <ref class="footnoteRef" idref="fn1">1</ref><note type="footnote" id="fn1"><num>1</num> So in original.</note> (or an amendment made by this title); <sup>1</sup></content>
<content id="c2">the Bacon-Davis Act,<sup>2</sup> in determining the time.</content>
<content id="c3">a vapor pressure less than 10−<sup>4</sup> Pa.</content>
</section>
</title></main></uscDoc>
"""


def test_repeated_footnote_sup_removed_exponent_kept():
    # Found in 29 U.S.C. 218c(a)(2) at release point 119-121: a footnote
    # cited again is marked by a bare <sup>, not <ref class="footnoteRef">.
    d = Path(tempfile.mkdtemp())
    p = d / "usc98.xml"
    p.write_text(SUP_FIXTURE, encoding="utf-8")
    _, units = extract_title(p)
    texts = dict(zip(units["parent_id"], units["text_display"]))
    assert texts["c1"] == ("any provision of this title (or an amendment "
                           "made by this title);")
    assert texts["c2"] == "the Bacon-Davis Act, in determining the time."
    assert texts["c3"] == "a vapor pressure less than 10−4 Pa."


def check_real_titles(*paths):
    """Smoke check on downloaded titles, e.g. usc29.xml and usc18.xml."""
    for p in paths:
        elements, units = extract_title(p)
        print(Path(p).name, elements["elem_source"].value_counts().to_dict(),
              "units:", len(units))
        if "usc18" in Path(p).name:
            hit = units[units["identifier"].fillna("").str.startswith(
                "/us/usc/t18/s3553/f")]
            assert (hit["elem_source"] == "chapeau_condensed").any(), \
                "18 U.S.C. 3553(f) should be one condensed chapeau unit"
            print(hit.iloc[0]["text_parse"][:400])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    if len(sys.argv) > 1:
        check_real_titles(*sys.argv[1:])
