import pandas as pd
import pytest

from negconj.shards import merge_shards, plan_shards, prefilter, read_done, run_shard

from test_segment import split_tokenizer


class FakeParser:
    stanza_version = "fake"
    model_package = "fake"

    def __init__(self):
        self.calls = 0

    def parse(self, token_lists):
        self.calls += 1
        res = [("(S " + " ".join(f"(X {t})" for t in toks) + ")", ["X"] * len(toks))
               for toks in token_lists]
        # one sentence always fails, to exercise the failure path
        errs = [None] * len(res)
        for i, toks in enumerate(token_lists):
            if "unparseable" in toks:
                res[i], errs[i] = None, "boom"
        return res, errs


def make_units(n_sections=12):
    rows = []
    for s in range(n_sections):
        for u in range(3):
            text = (f"Section {s} unit {u} shall not apply to a and b. "
                    f"Other text here and there. It is not x and y.")
            if s == 5 and u == 1:
                text = "This is unparseable and not ok and fine. Then not this and that."
            rows.append({"unit_id": f"id{s}x{u}_1", "title": "99", "is_appendix": False,
                         "section": f"/us/usc/t99/s{s}" if s != 11 else None,
                         "identifier": f"/us/usc/t99/s{s}", "parent_id": f"id{s}x{u}",
                         "parent_position": 1, "elem_source": "content",
                         "text_display": text, "text_parse": text, "n_elements": 1})
    rows.append({**rows[0], "unit_id": "nomatch_1", "text_parse": "Nothing to see.",
                 "text_display": "Nothing to see."})
    return pd.DataFrame(rows)


def test_prefilter_and_section_keys():
    pre = prefilter(make_units())
    assert "nomatch_1" not in set(pre["unit_id"])
    assert "99/_nosection" in set(pre["section_key"])


def test_plan_deterministic_and_complete():
    pre = prefilter(make_units())
    p1 = plan_shards({"99": pre}, target_chars=600)
    p2 = plan_shards({"99": pre.copy()}, target_chars=600)
    assert p1 == p2 and len(p1) > 1
    flat = [s for v in p1.values() for s in v]
    assert flat == list(dict.fromkeys(pre["section_key"]))  # every section once, in order


def _run_all(root, plan, pre, crash=None):
    parser = FakeParser()
    for sid, secs in plan.items():
        run_shard(sid, secs, pre, root, split_tokenizer, parser, min_matches=4, seg_chunk=3,
                  hook=crash or (lambda stage, k: None))
    return merge_shards(root, plan, pre)


@pytest.mark.parametrize("crash_stage,crash_k", [("part", 1), ("done", 2), ("part", 0)])
def test_checkpoint_resume_equals_uninterrupted(tmp_path, crash_stage, crash_k):
    pre = prefilter(make_units())
    plan = {"99-000": list(dict.fromkeys(pre["section_key"]))}

    ref_root = tmp_path / "ref"
    ref_counts = _run_all(ref_root, plan, pre)
    ref = pd.read_parquet(ref_root / "results" / "parses.parquet")

    root = tmp_path / "crash"
    fired = []

    def crash(stage, k):
        if (stage, k) == (crash_stage, crash_k) and not fired:
            fired.append(1)
            raise RuntimeError("preempted")

    with pytest.raises(RuntimeError, match="preempted"):
        run_shard("99-000", plan["99-000"], pre, root, split_tokenizer, FakeParser(),
                  min_matches=4, seg_chunk=3, hook=crash)
    n_done = len(read_done(root, "99-000"))
    assert 0 <= n_done < len(plan["99-000"])
    counts = _run_all(root, plan, pre)
    got = pd.read_parquet(root / "results" / "parses.parquet")

    assert got["sent_id"].is_unique
    assert got["sent_id"].tolist() == ref["sent_id"].tolist()
    pd.testing.assert_frame_equal(got, ref)
    assert counts["matches"] == ref_counts["matches"]
    assert counts["failures"] == ref_counts["failures"] == 1
    if crash_stage == "part":
        assert counts["duplicate_parse_rows_dropped"] > 0


def test_merge_detects_unfinished_shard(tmp_path):
    pre = prefilter(make_units())
    plan = {"99-000": list(dict.fromkeys(pre["section_key"]))}
    with pytest.raises(AssertionError, match="not done"):
        merge_shards(tmp_path, plan, pre)
