import pandas as pd
import pytest

from negconj.classify import (
    and_ccs, c2, classify_frame, classify_tree, dominates, n_conjuncts,
    not_nodes, to_tree, tree_distance,
)

# The table in work plan §6: (tree, C2 strict, C2 relaxed).
PLAN_TABLE = [
    ("(S (NP X) (VP (MD shall) (RB not) (VP (VB have) (NP (NP b) (CC and) (NP c)))))", True, True),
    ("(S (NP X) (VP (MD shall) (RB not) (VP (VP (VB sell)) (CC and) (VP (VB buy)))))", True, True),
    ("(S (NP X) (VP (MD shall) (ADVP (RB not)) (VP (VB have) (NP (NP b) (CC and) (NP c)))))", False, True),
    ("(S (NP X) (VP (MD shall) (RB not) (VP (VB act))) (PP (IN except) (PP (IN in) (NP (NP A) (CC and) (NP B)))))", False, False),
    ("(S (S (NP X) (VP (VBZ is) (RB not) (ADJP red))) (CC and) (S (NP Y) (VP (VBZ is) (ADJP blue))))", False, False),
]


@pytest.mark.parametrize("tree_str,strict,relaxed", PLAN_TABLE)
def test_plan_table(tree_str, strict, relaxed):
    t = to_tree(tree_str)
    [nn] = not_nodes(t)
    [cc] = and_ccs(t)
    coord = cc.parent()
    assert c2(nn, coord) is strict
    assert c2(nn, coord, relaxed=True) is relaxed
    _, summ = classify_tree("x", tree_str)
    assert summ["flagged"] is strict and summ["flagged_relaxed"] is relaxed


def test_dominates_is_proper():
    t = to_tree("(S (NP a) (VP (VB b)))")
    vp = t[1]
    assert dominates(t, vp) and not dominates(vp, vp) and not dominates(vp, t)


def test_pair_columns_pulsifer_shape():
    rows, summ = classify_tree("p", "(ROOT " + PLAN_TABLE[0][0] + ")")
    [r] = rows
    assert r["not_parent_label"] == "VP" and r["coord_label"] == "NP"
    assert r["n_conjuncts"] == 2 and r["not_before_and"]
    # RB -> VP -> VP -> NP(coord): 3 edges
    assert r["tree_distance"] == 3
    assert summ["n_pairs"] == 1


def test_case_insensitive_and_multiple_pairs():
    s = ("(S (NP X) (VP (MD shall) (RB Not) (VP (VB have) (NP (NP a) (, ,) (NP b) (, ,) (CC AND) (NP c)))) "
         "(CC and) (S (NP Y) (VP (VBZ does) (RB not) (VP (VB go)))))")
    rows, summ = classify_tree("m", s)
    assert summ["n_not"] == 2 and summ["n_and_cc"] == 2 and len(rows) == 4
    first = [r for r in rows if r["not_index"] == 2]
    inner = [r for r in first if r["coord_label"] == "NP"][0]
    assert inner["c2_strict"] and inner["n_conjuncts"] == 3
    # the sentence-level S coordination is not dominated by either VP
    assert not any(r["c2_strict"] for r in rows if r["coord_label"] == "S")
    assert summ["flagged"]


def test_no_and_cc_means_not_flagged():
    # "and" mis-tagged, or "and/or" as one token: no coordination node.
    rows, summ = classify_tree("n", "(S (NP X) (VP (MD shall) (RB not) (VP (VB act) (NP (NN and/or)))))")
    assert rows == [] and not summ["flagged"] and summ["n_not"] == 1


def test_tree_distance_symmetric():
    t = to_tree(PLAN_TABLE[3][0])
    [nn] = not_nodes(t)
    coord = and_ccs(t)[0].parent()
    # RB -> VP -> S, then S -> PP -> PP -> NP
    assert tree_distance(nn, coord) == tree_distance(coord, nn) == 5
    assert n_conjuncts(coord) == 2


def test_classify_frame_keeps_all_rows():
    parses = pd.DataFrame({"sent_id": [f"s{i}" for i in range(5)],
                           "tree": [t for t, _, _ in PLAN_TABLE]})
    pairs, sents = classify_frame(parses)
    assert len(sents) == 5 and len(pairs) == 5
    assert sents["flagged"].tolist() == [s for _, s, _ in PLAN_TABLE]
    assert sents["flagged_relaxed"].tolist() == [r for _, _, r in PLAN_TABLE]


def test_or_connective_mirrors_and():
    """The disjunction run uses the same C2 definitions with CC "or"."""
    for tree_str, strict, relaxed in PLAN_TABLE:
        or_tree = tree_str.replace("(CC and)", "(CC or)")
        rows, summ = classify_tree("o", or_tree, conn="or")
        assert summ["flagged"] is strict and summ["flagged_relaxed"] is relaxed
        assert summ["n_or_cc"] == 1 and "or_index" in rows[0] and rows[0]["not_before_or"]
        # an "or" tree has no (not, and) pairs, and vice versa
        assert classify_tree("o", or_tree, conn="and")[1]["n_pairs"] == 0
        assert classify_tree("a", tree_str, conn="or")[1]["n_pairs"] == 0


def test_classify_frame_or_columns():
    parses = pd.DataFrame({"sent_id": ["s0"], "tree": [PLAN_TABLE[0][0].replace("(CC and)", "(CC or)")]})
    pairs, sents = classify_frame(parses, conn="or")
    assert list(pairs.columns)[2:4] == ["or_index", "not_before_or"]
    assert "n_or_cc" in sents and bool(sents["flagged"].iloc[0])
