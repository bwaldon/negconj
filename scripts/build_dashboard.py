"""Build a self-contained local HTML viewer for classified sentences and their trees.

    python scripts/build_dashboard.py results_pilot            # -> results_pilot/dashboard.html
    python scripts/build_dashboard.py results --only-flagged   # smaller file for the full run

Open the output file directly in a browser; no server needed. Review marks are kept in
the browser's localStorage and can be exported as CSV from the page.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def build(results: Path, only_flagged: bool) -> Path:
    cls = pd.read_parquet(results / "sentences_classified.parquet")
    pairs = pd.read_parquet(results / "pairs.parquet")
    if only_flagged:
        cls = cls[cls["flagged"] | cls["flagged_relaxed"]]
    pairs = pairs[pairs["sent_id"].isin(cls["sent_id"])]
    pcols = ["not_index", "and_index", "not_parent_label", "not_relaxed_parent_label",
             "coord_label", "n_conjuncts", "tree_distance", "c2_strict", "c2_relaxed"]
    by_sent = {sid: g[pcols].values.tolist() for sid, g in pairs.groupby("sent_id", sort=False)}
    rows = []
    for r in cls.itertuples(index=False):
        rows.append([r.sent_id, str(r.title), r.identifier, r.elem_source, list(r.tokens), r.tree,
                     bool(r.flagged), bool(r.flagged_relaxed),
                     [[int(p[0]), int(p[1]), p[2], p[3], p[4], int(p[5]), int(p[6]), bool(p[7]), bool(p[8])]
                      for p in by_sent.get(r.sent_id, [])]])
    manifest = {}
    mp = results / "RUN_MANIFEST.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        manifest = {k: m.get(k) for k in ("release_point", "git_commit", "model_package", "gpu_type")}
        manifest["partial"] = m.get("PARTIAL")
    data = {"run": results.resolve().name, "manifest": manifest, "only_flagged": only_flagged,
            "n_total": int(len(pd.read_parquet(results / "sentences_classified.parquet", columns=["sent_id"]))),
            "rows": rows}
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = (HERE / "dashboard_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)
    out = results / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?", default=Path("results_pilot"))
    ap.add_argument("--only-flagged", action="store_true",
                    help="include only flagged (strict or relaxed) sentences")
    a = ap.parse_args()
    out = build(a.results, a.only_flagged)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
