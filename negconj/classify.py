"""Tree predicates and criterion C2 (work plan §6).

Trees are handled as ``nltk.tree.ParentedTree`` built from the stored bracketed
string, so nothing here depends on Stanza's tree API.

Definitions
-----------
not-node           a preterminal whose leaf lowercases to "not".
coordination node  for each preterminal CC whose leaf lowercases to "and", the
                   node immediately dominating it.
dominates(a, b)    a is a proper ancestor of b.
C2 strict          not's parent is labelled VP and dominates the coordination node.
C2 relaxed         as strict, after first skipping unary projections above not
                   (e.g. (ADVP (RB not))).
"""

from __future__ import annotations

import pandas as pd
from nltk.tree import ParentedTree

# Children of a coordination node that are not conjuncts.
NON_CONJUNCT_LABELS = frozenset({"CC", ",", ":", ".", "``", "''", "-LRB-", "-RRB-", "HYPH"})

PAIR_COLUMNS = [
    "sent_id", "not_index", "and_index", "not_before_and",
    "not_parent_label", "not_relaxed_parent_label", "coord_label",
    "n_conjuncts", "tree_distance", "c2_strict", "c2_relaxed",
]


def to_tree(s: str) -> ParentedTree:
    return ParentedTree.fromstring(s)


def is_preterminal(t) -> bool:
    return isinstance(t, ParentedTree) and len(t) == 1 and isinstance(t[0], str)


def preterminals(tree: ParentedTree) -> list[ParentedTree]:
    """Preterminals in leaf order."""
    return [st for st in tree.subtrees() if is_preterminal(st)]


def not_nodes(tree: ParentedTree) -> list[ParentedTree]:
    return [p for p in preterminals(tree) if p[0].lower() == "not"]


def and_ccs(tree: ParentedTree) -> list[ParentedTree]:
    return [p for p in preterminals(tree) if p.label() == "CC" and p[0].lower() == "and"]


def dominates(a, b) -> bool:
    p = b.parent()
    while p is not None:
        if p is a:
            return True
        p = p.parent()
    return False


def _relaxed_parent(not_node):
    p = not_node.parent()
    while p is not None and len(p) == 1:
        p = p.parent()
    return p


def c2(not_node, coord_node, relaxed: bool = False) -> bool:
    p = _relaxed_parent(not_node) if relaxed else not_node.parent()
    return p is not None and p.label().startswith("VP") and dominates(p, coord_node)


def n_conjuncts(coord_node) -> int:
    return sum(1 for ch in coord_node
               if not (isinstance(ch, ParentedTree) and ch.label() in NON_CONJUNCT_LABELS))


def tree_distance(a, b) -> int:
    """Number of edges on the path between nodes a and b."""
    pa, pb = a.treeposition(), b.treeposition()
    common = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        common += 1
    return (len(pa) - common) + (len(pb) - common)


def classify_tree(sent_id: str, tree_str: str) -> tuple[list[dict], dict]:
    """Return (pair rows, sentence summary) for one parse."""
    tree = to_tree(tree_str)
    pts = preterminals(tree)
    leaf_index = {id(p): i for i, p in enumerate(pts)}
    nots = [p for p in pts if p[0].lower() == "not"]
    ands = [p for p in pts if p.label() == "CC" and p[0].lower() == "and"]
    rows = []
    for nn in nots:
        parent = nn.parent()
        rparent = _relaxed_parent(nn)
        for cc in ands:
            coord = cc.parent()
            rows.append({
                "sent_id": sent_id,
                "not_index": leaf_index[id(nn)],
                "and_index": leaf_index[id(cc)],
                "not_before_and": leaf_index[id(nn)] < leaf_index[id(cc)],
                "not_parent_label": parent.label() if parent is not None else None,
                "not_relaxed_parent_label": rparent.label() if rparent is not None else None,
                "coord_label": coord.label(),
                "n_conjuncts": n_conjuncts(coord),
                "tree_distance": tree_distance(nn, coord),
                "c2_strict": c2(nn, coord),
                "c2_relaxed": c2(nn, coord, relaxed=True),
            })
    summary = {
        "sent_id": sent_id,
        "n_not": len(nots),
        "n_and_cc": len(ands),
        "n_pairs": len(rows),
        "flagged": any(r["c2_strict"] for r in rows),
        "flagged_relaxed": any(r["c2_relaxed"] for r in rows),
    }
    return rows, summary


def classify_frame(parses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify every row of a parses table (needs sent_id, tree).

    Returns (pairs, sentences_classified), where sentences_classified is the
    input with the sentence-level columns joined on.
    """
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for sid, tree in zip(parses["sent_id"], parses["tree"]):
        rows, summ = classify_tree(sid, tree)
        all_rows.extend(rows)
        summaries.append(summ)
    pairs = pd.DataFrame(all_rows, columns=PAIR_COLUMNS)
    summ = pd.DataFrame(summaries, columns=[
        "sent_id", "n_not", "n_and_cc", "n_pairs", "flagged", "flagged_relaxed"])
    sentences = parses.merge(summ, on="sent_id", how="left", validate="one_to_one")
    return pairs, sentences
