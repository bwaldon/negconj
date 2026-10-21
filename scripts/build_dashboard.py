"""Build the static dashboard for one or more classified runs (negative conjunction, disjunction).

    python scripts/build_dashboard.py --and results --or results_disj --out site
    python scripts/build_dashboard.py --and results_pilot --out site_pilot --all-matches

Writes ``{out}/index.html`` plus, per dataset, ``data/{conn}/index.js`` (sentences, flags and
(not, connective) pairs: everything filtering and search need) and ``data/{conn}/trees-{title}.js``
(trees, loaded when a sentence is opened). The files are plain scripts, so the site works from
GitHub Pages and when opened from disk. Review marks are kept in the browser's localStorage and
can be exported as CSV from the page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nltk.tree import ParentedTree

from negconj.classify import is_preterminal
from negconj.shards import title_sort_key

HERE = Path(__file__).resolve().parent
LABELS = {"and": "Negative conjunction", "or": "Negative disjunction"}


# An if-clause is an SBAR whose first word is "if" (also "only if", "even if"). A pair is inside
# one when some SBAR above not is an if-clause and also dominates (or is) the coordination node.
# Same rule as ifClauseOf() in the page, which draws the brackets once the tree is loaded.
def _if_words(leaves: list[str], lo: int) -> bool:
    a = leaves[lo].lower() if lo < len(leaves) else ""
    b = leaves[lo + 1].lower() if lo + 1 < len(leaves) else ""
    return a == "if" or (a in ("only", "even") and b == "if")


def in_if_flags(tree_str: str, pairs: list[tuple[int, int]]) -> list[bool]:
    """For each (not leaf index, connective leaf index), whether the pair sits in an if-clause."""
    tree = ParentedTree.fromstring(tree_str)
    pts = [st for st in tree.subtrees() if is_preterminal(st)]
    leaves = [p[0] for p in pts]
    first: dict[int, int] = {}  # id(node) -> index of its first leaf
    for i, p in enumerate(pts):
        n = p
        while n is not None and id(n) not in first:
            first[id(n)] = i
            n = n.parent()
    out = []
    for not_i, cc_i in pairs:
        above_coord = set()
        n = pts[cc_i].parent()
        while n is not None:
            above_coord.add(id(n))
            n = n.parent()
        a = pts[not_i].parent()
        while a is not None and not (a.label().startswith("SBAR") and id(a) in above_coord
                                     and _if_words(leaves, first[id(a)])):
            a = a.parent()
        out.append(a is not None)
    return out


def _js(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def _manifest(results: Path) -> dict:
    mp = results / "RUN_MANIFEST.json"
    if not mp.exists():
        return {}
    m = json.loads(mp.read_text())
    out = {k: m.get(k) for k in ("release_point", "git_commit", "model_package", "gpu_type")}
    out["partial"] = m.get("PARTIAL")
    return out


def build_dataset(conn: str, results: Path, out: Path, only_flagged: bool) -> dict:
    cls = pd.read_parquet(results / "sentences_classified.parquet")
    pairs = pd.read_parquet(results / "pairs.parquet")
    n_total = len(cls)
    if only_flagged:
        cls = cls[cls["flagged"]]
    pairs = pairs[pairs["sent_id"].isin(cls["sent_id"])]
    cc = f"{conn}_index"
    pcols = ["not_index", cc, "not_parent_label", "not_relaxed_parent_label",
             "coord_label", "n_conjuncts", "tree_distance", "c2_strict", "c2_relaxed"]
    by_sent = {sid: g[pcols].values.tolist() for sid, g in pairs.groupby("sent_id", sort=False)}
    rows, trees = [], {}
    n_if = 0
    for r in cls.itertuples(index=False):
        ps = by_sent.get(r.sent_id, [])
        ifs = in_if_flags(r.tree, [(int(p[0]), int(p[1])) for p in ps])
        n_if += any(f and p[7] for f, p in zip(ifs, ps))
        rows.append([r.sent_id, str(r.title), r.identifier, r.elem_source, " ".join(r.tokens),
                     bool(r.flagged), bool(r.flagged_relaxed),
                     [[int(p[0]), int(p[1]), p[2], p[3], p[4], int(p[5]), int(p[6]), bool(p[7]), bool(p[8]), f]
                      for p, f in zip(ps, ifs)]])
        trees.setdefault(str(r.title), {})[r.sent_id] = r.tree
    d = out / "data" / conn
    d.mkdir(parents=True, exist_ok=True)
    meta = {"conn": conn, "label": LABELS[conn], "run": results.resolve().name,
            "manifest": _manifest(results), "only_flagged": only_flagged, "n_total": n_total,
            "titles": sorted(trees, key=title_sort_key)}
    (d / "index.js").write_text(f"NEG.index({_js(conn)},{_js({**meta, 'rows': rows})});\n", encoding="utf-8")
    for title, tmap in trees.items():
        (d / f"trees-{title}.js").write_text(f"NEG.trees({_js(conn)},{_js(title)},{_js(tmap)});\n",
                                             encoding="utf-8")
    sizes = {p.name: p.stat().st_size for p in d.iterdir()}
    return {"conn": conn, "rows": len(rows), "flagged_with_strict_if_pair": n_if,
            "index_mb": sizes["index.js"] / 1e6, "trees_mb": sum(v for k, v in sizes.items()
                                                               if k.startswith("trees")) / 1e6}


def build(datasets: dict[str, Path], out: Path, only_flagged: bool) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    stats = [build_dataset(c, p, out, only_flagged) for c, p in datasets.items()]
    html = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATASETS__*/[]", _js(list(datasets)))
    (out / "index.html").write_text(html, encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--and", dest="and_", type=Path, help="results dir of the negative-conjunction run")
    ap.add_argument("--or", dest="or_", type=Path, help="results dir of the negative-disjunction run")
    ap.add_argument("--out", type=Path, default=Path("site"))
    ap.add_argument("--all-matches", action="store_true",
                    help="include unflagged matches too (default: flagged sentences, C2 strict)")
    a = ap.parse_args()
    datasets = {c: p for c, p in (("and", a.and_), ("or", a.or_)) if p is not None}
    if not datasets:
        ap.error("give --and and/or --or")
    for s in build(datasets, a.out, not a.all_matches):
        print(s)
    print(f"wrote {a.out / 'index.html'}")


if __name__ == "__main__":
    main()
