"""Prefilter, shard planning, checkpointed shard processing, merge (§5.3–5.5).

Data layout under a root directory (``/data`` on Modal, ``data/`` locally):

    units/{title}.parquet              extracted units (per title)
    shards.json                        shard ID -> ordered section keys
    progress/{shard}.done              section keys finished, one per line
    parses/{shard}/part-{k:05d}.parquet  parsed matches of batch k
    parses/{shard}/seg-{k:05d}.parquet   every segmented sentence of batch k
    failures/{shard}.jsonl             sentences that failed to parse
    results/                           merged outputs

A section counts as done only once the batch's part and seg files are
written; a shard restarted after preemption skips done sections and appends
new part files, so a batch can appear twice and merge drops the duplicates.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .segment import segment_units

NOT_RE = re.compile(r"\bnot\b")
AND_RE = re.compile(r"\band\b")

PARSE_COLUMNS = ["sent_id", "unit_id", "section", "title", "text", "tokens",
                 "xpos", "n_tokens", "tree", "stanza_version", "model_package"]
SEG_COLUMNS = ["sent_id", "unit_id", "section", "title", "n_tokens", "n_merged", "matched"]


# --------------------------------------------------------------------------
# Prefilter and planning
# --------------------------------------------------------------------------

def section_key(section, title: str) -> str:
    return section if isinstance(section, str) and section else f"{title}/_nosection"


def prefilter(units: pd.DataFrame) -> pd.DataFrame:
    """Units whose lowercased text_parse contains both 'not' and 'and' as
    words; adds ``section_key``."""
    low = units["text_parse"].str.lower()
    keep = low.str.contains(NOT_RE) & low.str.contains(AND_RE)
    out = units[keep].copy()
    out["section_key"] = [section_key(s, t) for s, t in zip(out["section"], out["title"])]
    return out


def title_sort_key(title: str) -> tuple[int, str]:
    m = re.match(r"(\d+)(.*)", title)
    return (int(m.group(1)), m.group(2)) if m else (10**6, title)


def plan_shards(prefiltered: dict[str, pd.DataFrame], target_chars: int) -> dict[str, list[str]]:
    """Group consecutive sections within each title into shards of about
    ``target_chars`` characters of prefiltered text (a GPU-time proxy).
    Deterministic: titles in numeric order, sections in document order."""
    plan: dict[str, list[str]] = {}
    for title in sorted(prefiltered, key=title_sort_key):
        df = prefiltered[title]
        if df.empty:
            continue
        sizes = df.groupby("section_key", sort=False)["text_parse"].apply(lambda s: s.str.len().sum())
        k, cur, cur_size = 0, [], 0
        for sec, sz in sizes.items():
            if cur and cur_size + sz > target_chars:
                plan[f"{title}-{k:03d}"] = cur
                k, cur, cur_size = k + 1, [], 0
            cur.append(sec)
            cur_size += int(sz)
        if cur:
            plan[f"{title}-{k:03d}"] = cur
    return plan


def shard_title(shard_id: str) -> str:
    return shard_id.rsplit("-", 1)[0]


# --------------------------------------------------------------------------
# Atomic file helpers
# --------------------------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_done(root: Path, shard_id: str) -> list[str]:
    p = root / "progress" / f"{shard_id}.done"
    if not p.exists():
        return []
    return [line for line in p.read_text(encoding="utf-8").splitlines() if line]


def write_done(root: Path, shard_id: str, keys: list[str]) -> None:
    atomic_write_bytes(root / "progress" / f"{shard_id}.done",
                       ("".join(k + "\n" for k in keys)).encode("utf-8"))


def _next_part(shard_dir: Path) -> int:
    ks = [int(m.group(1)) for p in shard_dir.glob("*.parquet")
          if (m := re.fullmatch(r"(?:part|seg)-(\d+)\.parquet", p.name))]
    return max(ks) + 1 if ks else 0


# --------------------------------------------------------------------------
# Shard processing
# --------------------------------------------------------------------------

def _batches(sections: list[str], sents_by_section: Callable[[str], pd.DataFrame],
             min_matches: int) -> Iterable[tuple[list[str], list[pd.DataFrame]]]:
    """Consecutive sections until they hold >= min_matches matched sentences.
    A section is never split across batches."""
    keys, frames, n = [], [], 0
    for sec in sections:
        df = sents_by_section(sec)
        keys.append(sec)
        frames.append(df)
        n += int(df["matched"].sum())
        if n >= min_matches:
            yield keys, frames
            keys, frames, n = [], [], 0
    if keys:
        yield keys, frames


def run_shard(shard_id: str, sections: list[str], units: pd.DataFrame, root: Path,
              tokenizer, parser, *, min_matches: int = 32, seg_chunk: int = 64,
              commit: Callable[[], None] = lambda: None,
              hook: Callable[[str, int], None] = lambda stage, k: None) -> dict:
    """Process one shard; resumable. ``units`` are the prefiltered units of
    the shard's title (with section_key). ``parser`` needs .parse(token_lists)
    -> (results, errors), .stanza_version, .model_package. ``hook(stage, k)``
    is called after 'part' and 'done' of batch k (tests use it to crash)."""
    t0 = time.monotonic()
    tok_stats0 = dict(getattr(tokenizer, "stats", {}))
    root = Path(root)
    shard_dir = root / "parses" / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = read_done(root, shard_id)
    done_set = set(done)
    todo = [s for s in sections if s not in done_set]
    units = units[units["section_key"].isin(set(todo))]
    by_sec = {k: g for k, g in units.groupby("section_key", sort=False)}

    # Segment lazily, seg_chunk sections at a time (one tokenizer call each).
    cache: dict[str, pd.DataFrame] = {}
    seg_stats: dict[str, int] = {}
    empty = units.iloc[0:0]

    def sents_of(sec: str) -> pd.DataFrame:
        if sec not in cache:
            i = todo.index(sec)
            chunk = [s for s in todo[i:i + seg_chunk] if s not in cache]
            u = pd.concat([by_sec.get(s, empty) for s in chunk])
            sents, st = segment_units(u, tokenizer)
            for k, v in st.items():
                seg_stats[k] = seg_stats.get(k, 0) + v
            for s in chunk:
                cache[s] = sents[sents["section"] == s]
        return cache.pop(sec)

    counts = {"sections": 0, "batches": 0, "sentences": 0, "matches": 0,
              "parsed": 0, "failed": 0}
    k = _next_part(shard_dir)
    for keys, frames in _batches(todo, sents_of, min_matches):
        sents = pd.concat(frames, ignore_index=True)
        matched = sents[sents["matched"]].reset_index(drop=True)
        results, errors = parser.parse(list(matched["tokens"]))
        rows, fails = [], []
        for r, (row, res, err) in enumerate(zip(matched.itertuples(index=False), results, errors)):
            if res is None:
                fails.append({"sent_id": row.sent_id, "unit_id": row.unit_id,
                              "section": row.section, "error": err})
                continue
            tree, xpos = res
            rows.append({"sent_id": row.sent_id, "unit_id": row.unit_id,
                         "section": row.section, "title": row.title, "text": row.text,
                         "tokens": list(row.tokens), "xpos": xpos, "n_tokens": row.n_tokens,
                         "tree": tree, "stanza_version": parser.stanza_version,
                         "model_package": parser.model_package})
        if fails:
            fpath = root / "failures" / f"{shard_id}.jsonl"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            prev = fpath.read_bytes() if fpath.exists() else b""
            atomic_write_bytes(fpath, prev + "".join(json.dumps(f) + "\n" for f in fails).encode())
        atomic_write_parquet(pd.DataFrame(rows, columns=PARSE_COLUMNS),
                             shard_dir / f"part-{k:05d}.parquet")
        atomic_write_parquet(sents[SEG_COLUMNS], shard_dir / f"seg-{k:05d}.parquet")
        hook("part", k)
        done.extend(keys)
        write_done(root, shard_id, done)
        commit()
        hook("done", k)
        counts["sections"] += len(keys)
        counts["batches"] += 1
        counts["sentences"] += len(sents)
        counts["matches"] += len(matched)
        counts["parsed"] += len(rows)
        counts["failed"] += len(fails)
        k += 1
    counts["seconds"] = time.monotonic() - t0
    counts.update({f"seg_{k2}": v for k2, v in seg_stats.items()})
    counts["parser_events"] = dict(getattr(parser, "events", {}))
    counts["tokenizer_stats"] = {k: v - tok_stats0.get(k, 0)
                                 for k, v in getattr(tokenizer, "stats", {}).items()}
    return counts


# --------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------

def _dedup(df: pd.DataFrame, what: str) -> tuple[pd.DataFrame, int]:
    dup = df["sent_id"].duplicated(keep=False)
    if not dup.any():
        return df, 0
    def norm(v):
        return tuple(v.tolist()) if hasattr(v, "tolist") else tuple(v) if isinstance(v, list) else v

    seen: dict[str, tuple] = {}
    for row in df[dup].itertuples(index=False):
        key = tuple(norm(v) for v in row)
        first = seen.setdefault(row.sent_id, key)
        if first != key:
            raise AssertionError(f"{what}: non-identical duplicate rows for {row.sent_id}")
    out = df.drop_duplicates("sent_id", keep="first")
    return out, len(df) - len(out)


def merge_shards(root: Path, plan: dict[str, list[str]], prefiltered_units: pd.DataFrame) -> dict:
    """Merge part/seg files into results/parses.parquet and
    results/segments.parquet, with the §5.5 assertions. Returns counts."""
    root = Path(root)
    parts, segs, fails = [], [], []
    for shard_id, sections in plan.items():
        done = set(read_done(root, shard_id))
        missing = [s for s in sections if s not in done]
        if missing:
            raise AssertionError(f"{shard_id}: {len(missing)} planned sections not done, "
                                 f"e.g. {missing[0]}")
        d = root / "parses" / shard_id
        parts += [pd.read_parquet(p) for p in sorted(d.glob("part-*.parquet"))]
        segs += [pd.read_parquet(p) for p in sorted(d.glob("seg-*.parquet"))]
        fp = root / "failures" / f"{shard_id}.jsonl"
        if fp.exists():
            fails += [json.loads(l) for l in fp.read_text().splitlines() if l]
    parses = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PARSE_COLUMNS)
    segments = pd.concat(segs, ignore_index=True) if segs else pd.DataFrame(columns=SEG_COLUMNS)
    parses, dup_parses = _dedup(parses, "parses")
    segments, dup_segs = _dedup(segments, "segments")
    failures = pd.DataFrame(fails, columns=["sent_id", "unit_id", "section", "error"])
    failures = failures.drop_duplicates("sent_id", keep="last")
    failures = failures[~failures["sent_id"].isin(parses["sent_id"])]

    missing_units = set(prefiltered_units["unit_id"]) - set(segments["unit_id"])
    if missing_units:
        raise AssertionError(f"{len(missing_units)} prefiltered units never segmented, "
                             f"e.g. {sorted(missing_units)[0]}")
    matched = set(segments.loc[segments["matched"].astype(bool), "sent_id"])
    covered = set(parses["sent_id"]) | set(failures["sent_id"])
    if matched != covered:
        raise AssertionError(f"matches without parse/failure: {len(matched - covered)}; "
                             f"parses without match: {len(covered - matched)}")

    order = {u: i for i, u in enumerate(prefiltered_units["unit_id"])}

    def sort(df):
        key = df["sent_id"].map(lambda s: (order.get(s.rsplit("_", 1)[0], -1), int(s.rsplit("_", 1)[1])))
        return df.assign(_k=key).sort_values("_k").drop(columns="_k").reset_index(drop=True)

    parses, segments = sort(parses), sort(segments)
    out = root / "results"
    atomic_write_parquet(parses, out / "parses.parquet")
    atomic_write_parquet(segments, out / "segments.parquet")
    atomic_write_parquet(failures, out / "failures.parquet")
    merges = int(segments["n_merged"].sum())
    return {"raw_sentence_splits": len(segments) + merges, "re_merged_splits": merges,
            "segmented_sentences": len(segments), "matches": len(matched),
            "parses": len(parses), "failures": len(failures),
            "duplicate_parse_rows_dropped": dup_parses,
            "duplicate_seg_rows_dropped": dup_segs}
