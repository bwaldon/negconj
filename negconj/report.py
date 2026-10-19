"""Consistency checks, review sample, summary, manifest and checksums (§7, §8)."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import pandas as pd

from .shards import read_done, title_sort_key

PULSIFER = "/us/usc/t18/s3553/f"
REVIEW_SEED = 20261019

LIMITATIONS = """\
- **Regex recall.** Only sentences matching `\\bnot\\b.*\\band\\b` are found: no `n't`, `cannot`,
  `nor`, `without`, `no`, `neither`, and no `and` that precedes the negation.
- **Coordination = CC "and".** A coordination node is the parent of a `CC` preterminal whose word is
  `and`. `and/or` (one token), `as well as`, and an `and` mis-tagged by the parser are not
  coordinations here.
- **Parser errors**, especially on very long condensed-chapeau sentences (hundreds of tokens),
  which are far outside the parser's training distribution. Stanza returns one best tree;
  no alternative parses are considered.
- **Syntactic, not interpretive.** `flagged` says the VP that is *not*'s parent dominates the
  coordination. It is not a judgment about which scope reading is correct in context, which,
  as the Campos-Chaves amicus brief stresses, depends on context.
- **Segmentation.** Stanza's sentence splitter plus a re-merge post-pass; residual splitting
  errors can split or join sentences.
"""


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def consistency_checks(root: Path, plan: dict[str, list[str]]) -> dict:
    root = Path(root)
    res = root / "results"
    segs = pd.read_parquet(res / "segments.parquet")
    parses = pd.read_parquet(res / "parses.parquet")
    fails = pd.read_parquet(res / "failures.parquet")
    cls = pd.read_parquet(res / "sentences_classified.parquet")
    checks: dict[str, bool | int] = {}

    matched = set(segs.loc[segs["matched"].astype(bool), "sent_id"])
    checks["every_match_has_one_parse_or_failure"] = (
        parses["sent_id"].is_unique and fails["sent_id"].is_unique
        and not (set(parses["sent_id"]) & set(fails["sent_id"]))
        and matched == set(parses["sent_id"]) | set(fails["sent_id"]))
    checks["no_duplicate_sent_ids"] = bool(segs["sent_id"].is_unique and parses["sent_id"].is_unique
                                           and cls["sent_id"].is_unique)
    checks["classified_equals_parses"] = set(cls["sent_id"]) == set(parses["sent_id"])
    checks["all_planned_sections_done"] = all(
        set(secs) <= set(read_done(root, sid)) for sid, secs in plan.items())

    # No text both standalone and inside a condensed chapeau.
    overlap = 0
    for p in sorted((root / "elements").glob("*.parquet")):
        el = pd.read_parquet(p, columns=["parent_id", "parent_position", "consumed_by"])
        un = pd.read_parquet(root / "units" / p.name, columns=["unit_id", "elem_source"])
        standalone = set(un.loc[un["elem_source"] != "chapeau_condensed", "unit_id"])
        consumed = el[el["consumed_by"].notna()]
        ids = consumed["parent_id"] + "_" + consumed["parent_position"].astype(str)
        overlap += int(ids.isin(standalone).sum())
    checks["standalone_and_condensed_overlap"] = overlap
    checks["ok"] = all(v for k, v in checks.items() if isinstance(v, bool)) and overlap == 0
    return checks


# --------------------------------------------------------------------------
# Review sample
# --------------------------------------------------------------------------

def _allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Proportional allocation with largest remainders, capped at stratum size."""
    n = min(n, int(sizes.sum()))
    exact = sizes / sizes.sum() * n
    alloc = exact.astype(int)
    rem = (exact - alloc).sort_values(ascending=False, kind="stable")
    for k in rem.index:
        if alloc.sum() >= n:
            break
        if alloc[k] < sizes[k]:
            alloc[k] += 1
    return alloc


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    strata = df.groupby(["title", "elem_source"], sort=True)
    alloc = _allocate(strata.size(), n)
    rng = random.Random(seed)
    picks = []
    for key, g in strata:
        k = int(alloc[key])
        if k:
            picks.append(g.loc[sorted(rng.sample(list(g.index), k))])
    return pd.concat(picks)


def review_sample(cls: pd.DataFrame, n_each: int = 100, seed: int = REVIEW_SEED) -> pd.DataFrame:
    cols = ["sent_id", "title", "section", "identifier", "elem_source", "flagged",
            "flagged_relaxed", "text", "text_display", "tree"]
    parts = [stratified_sample(cls[cls["flagged"]], n_each, seed),
             stratified_sample(cls[~cls["flagged"]], n_each, seed + 1)]
    out = pd.concat(parts)[cols].reset_index(drop=True)
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)  # blind the order
    for c in ("reviewer_is_negconj", "reviewer_scope_reading", "notes"):
        out[c] = ""
    return out


# --------------------------------------------------------------------------
# Summary, manifest, checksums
# --------------------------------------------------------------------------

def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def pulsifer_rows(cls: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = cls[cls["identifier"] == PULSIFER]
    return p, pairs[pairs["sent_id"].isin(p["sent_id"])]


def summary_md(stage_counts: dict, cls: pd.DataFrame, pairs: pd.DataFrame,
               fails: pd.DataFrame, checks: dict) -> str:
    by_title = (cls.groupby("title")
                .agg(matches=("sent_id", "size"), flagged=("flagged", "sum"),
                     flagged_relaxed=("flagged_relaxed", "sum")))
    by_title = by_title.loc[sorted(by_title.index, key=title_sort_key)].reset_index()
    flagged_pairs = pairs[pairs["c2_strict"]]
    by_label = (flagged_pairs.groupby("coord_label")["sent_id"].nunique()
                .sort_values(ascending=False).rename("flagged_sentences").reset_index())
    psent, ppairs = pulsifer_rows(cls, pairs)
    out = ["# Negative conjunctions in the U.S. Code: run summary", ""]
    out += ["## Stage counts", "", _md_table(pd.DataFrame(
        list(stage_counts.items()), columns=["stage", "count"])), ""]
    out += ["## Flagged sentences by title", "", _md_table(by_title), ""]
    out += ["## Flagged sentences by coordination label (C2 strict)", "",
            "A sentence is counted under every label of a coordination that makes it flagged.", "",
            _md_table(by_label), ""]
    out += ["## Pulsifer: 18 U.S.C. § 3553(f)", ""]
    if psent.empty:
        out += ["**Not found in the classified table.**", ""]
    for r in psent.itertuples():
        pp = ppairs[ppairs["sent_id"] == r.sent_id]
        out += [f"- `{r.sent_id}` ({r.n_tokens} tokens): flagged = **{r.flagged}**, "
                f"flagged_relaxed = **{r.flagged_relaxed}**, (not, and) pairs = {len(pp)}, "
                f"strict pairs = {int(pp['c2_strict'].sum())}", "",
                f"  > {r.text[:600]}{'…' if len(r.text) > 600 else ''}", ""]
    out += ["## Parse failures", "", f"{len(fails)} sentences failed to parse.", ""]
    if len(fails):
        out += [_md_table(fails[["sent_id", "error"]].head(50)), ""]
    out += ["## Consistency checks", "", _md_table(pd.DataFrame(
        list(checks.items()), columns=["check", "result"])), ""]
    out += ["## Known limitations", "", LIMITATIONS]
    return "\n".join(out)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(res: Path) -> Path:
    lines = []
    for p in sorted(res.rglob("*")):
        if p.is_file() and p.name != "CHECKSUMS.sha256" and not p.name.startswith("."):
            lines.append(f"{sha256_file(p)}  {p.relative_to(res).as_posix()}")
    out = res / "CHECKSUMS.sha256"
    out.write_text("\n".join(lines) + "\n")
    return out


def report_stage(root: Path, plan: dict[str, list[str]], manifest: dict,
                 stage_counts: dict) -> dict:
    """Write review_sample.csv, summary.md, RUN_MANIFEST.json, copy units/
    and elements/ into results/, then CHECKSUMS.sha256 (last)."""
    root = Path(root)
    res = root / "results"
    cls = pd.read_parquet(res / "sentences_classified.parquet")
    pairs = pd.read_parquet(res / "pairs.parquet")
    fails = pd.read_parquet(res / "failures.parquet")
    checks = consistency_checks(root, plan)

    review_sample(cls).to_csv(res / "review_sample.csv", index=False)
    (res / "summary.md").write_text(summary_md(stage_counts, cls, pairs, fails, checks))
    for sub in ("units", "elements"):
        (res / sub).mkdir(exist_ok=True)
        for p in (root / sub).glob("*.parquet"):
            dst = res / sub / p.name
            if not dst.exists() or dst.stat().st_size != p.stat().st_size:
                shutil.copy2(p, dst)
    manifest = {**manifest, "counts": stage_counts, "checks": checks}
    (res / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str))
    write_checksums(res)
    return checks
