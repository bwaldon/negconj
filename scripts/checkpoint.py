"""Phase-1 checkpoint material from a local run (data/local by default).

    python scripts/checkpoint.py [--root data/local] [--n 10] [--seed 0]

Prints per-title stage counts, the Pulsifer tree with its C2 result, and N random
flagged sentences with their trees; also runs report_stage to write
results/summary.md, review_sample.csv, RUN_MANIFEST.json and CHECKSUMS.sha256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nltk.tree import Tree

from negconj.report import PULSIFER, report_stage
from negconj.shards import prefilter


def pretty(tree: str, width: int = 110) -> str:
    return Tree.fromstring(tree).pformat(margin=width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/local"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, res = a.root, a.root / "results"
    plan = json.loads((root / "shards.json").read_text())
    segs = pd.read_parquet(res / "segments.parquet")
    cls = pd.read_parquet(res / "sentences_classified.parquet")
    pairs = pd.read_parquet(res / "pairs.parquet")
    fails = pd.read_parquet(res / "failures.parquet")

    rows = []
    for t in sorted(cls["title"].unique()):
        u = pd.read_parquet(root / "units" / f"{t}.parquet")
        pre = prefilter(u)
        s = segs[segs["title"] == t]
        c = cls[cls["title"] == t]
        rows.append({"title": t, "units": len(u),
                     "units_condensed": int((u["elem_source"] == "chapeau_condensed").sum()),
                     "prefilter_units": len(pre),
                     "prefilter_sections": pre["section_key"].nunique(),
                     "shards": sum(k.rsplit("-", 1)[0] == t for k in plan),
                     "raw_splits": len(s) + int(s["n_merged"].sum()),
                     "re_merged": int(s["n_merged"].sum()),
                     "sentences": len(s), "regex_matches": int(s["matched"].sum()),
                     "parsed": len(c),
                     "failed": int(fails["unit_id"].isin(pre["unit_id"]).sum()),
                     "pairs": int(pairs["sent_id"].isin(c["sent_id"]).sum()),
                     "flagged": int(c["flagged"].sum()),
                     "flagged_relaxed": int(c["flagged_relaxed"].sum())})
    counts = pd.DataFrame(rows).set_index("title").T
    print("== Stage counts\n" + counts.to_string() + "\n")

    p = cls[cls["identifier"] == PULSIFER]
    print("== Pulsifer (18 U.S.C. 3553(f))")
    for r in p.itertuples():
        pp = pairs[pairs["sent_id"] == r.sent_id]
        print(f"{r.sent_id}: {r.n_tokens} tokens; flagged={r.flagged} relaxed={r.flagged_relaxed}; "
              f"{len(pp)} pairs, {int(pp['c2_strict'].sum())} strict")
        print(r.text, "\n")
        print(pp.drop(columns="sent_id").to_string(), "\n")
        print(pretty(r.tree), "\n")

    flagged = cls[cls["flagged"]]
    pick = flagged.sample(min(a.n, len(flagged)), random_state=a.seed)
    print(f"== {len(pick)} random flagged sentences (seed {a.seed})")
    for r in pick.itertuples():
        pp = pairs[(pairs["sent_id"] == r.sent_id) & pairs["c2_strict"]]
        print(f"--- {r.identifier} [{r.elem_source}] {r.sent_id}")
        print(r.text)
        print("strict pairs:", pp[["not_index", "and_index", "coord_label", "n_conjuncts"]]
              .to_dict("records"))
        print(pretty(r.tree), "\n")

    stage_counts = {f"{k}_t{t}": int(v) for t in counts.columns for k, v in counts[t].items()}
    checks = report_stage(root, plan, {"run": "local-cpu", "release_point": "119-121"}, stage_counts)
    print("== report checks", checks)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
