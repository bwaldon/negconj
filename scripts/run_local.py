"""Run the pipeline locally on CPU (Phase 1), mirroring the Modal stages.

    python scripts/run_local.py --titles 29 18                  # full local run
    python scripts/run_local.py --smoke 50 --titles 29          # smoke test only

Outputs go under --root (default data/local), in the same layout as /data.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

from negconj.classify import classify_frame
from negconj.parse import Parser
from negconj.segment import build_tokenizer, segment_units, stanza_tokenizer
from negconj.shards import run_shard, shard_title
from negconj.stages import (classify_stage, extract_stage, load_prefiltered, merge_stage,
                            plan_stage)


def smoke(root: Path, title: str, n: int, seed: int = 0) -> None:
    pre = load_prefiltered(root, [title])[title]
    sents, stats = segment_units(pre, stanza_tokenizer(build_tokenizer()))
    m = sents[sents["matched"]].reset_index(drop=True)
    print(f"title {title}: prefiltered units {len(pre)}, sentences {len(sents)}, "
          f"matches {len(m)}, seg stats {stats}")
    pick = m.loc[sorted(random.Random(seed).sample(range(len(m)), min(n, len(m))))]
    parser = Parser(use_gpu=False)
    t = time.monotonic()
    results, errors = parser.parse(list(pick["tokens"]))
    dt = time.monotonic() - t
    ok = [r is not None for r in results]
    print(f"parsed {sum(ok)}/{len(pick)} in {dt:.1f}s ({dt / len(pick):.2f}s/sent, "
          f"{pick['n_tokens'].sum() / dt:.0f} tok/s); events {parser.events}")
    print("stanza", parser.stanza_version, "|", parser.model_package)
    parses = pick.assign(tree=[r[0] if r else None for r in results])[ok]
    pairs, cls = classify_frame(parses[["sent_id", "tree"]])
    print(f"pairs {len(pairs)}; flagged {int(cls['flagged'].sum())}, "
          f"relaxed {int(cls['flagged_relaxed'].sum())} of {len(cls)}")
    print(pairs.head(5).to_string())


def full(root: Path, xml_dir: Path, titles: list[str], target_chars: int, min_matches: int) -> None:
    log = {}
    for t in titles:
        log[f"extract_{t}"] = extract_stage(xml_dir / f"usc{t}.xml", root)
        print(log[f"extract_{t}"], flush=True)
    plan = plan_stage(root, target_chars, titles)
    print(f"plan: {len(plan)} shards", {t: sum(shard_title(s) == t for s in plan) for t in titles})
    pre = load_prefiltered(root, titles)
    tok = stanza_tokenizer(build_tokenizer())
    parser = Parser(use_gpu=False)
    for sid, secs in plan.items():
        c = run_shard(sid, secs, pre[shard_title(sid)], root, tok, parser, min_matches=min_matches)
        log[f"shard_{sid}"] = c
        print(sid, {k: c[k] for k in ("sections", "matches", "parsed", "failed", "seconds", "tokenizer_stats")},
              flush=True)
    log["merge"] = merge_stage(root, plan)
    print("merge", log["merge"])
    log["classify"] = classify_stage(root)
    print("classify", log["classify"])
    (root / "results" / "local_run_log.json").write_text(json.dumps(log, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/local"))
    ap.add_argument("--xml-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--titles", nargs="+", default=["29", "18"])
    ap.add_argument("--target-chars", type=int, default=400_000)
    ap.add_argument("--min-matches", type=int, default=256)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    if a.smoke:
        for t in a.titles:
            extract_stage(a.xml_dir / f"usc{t}.xml", a.root)
            smoke(a.root, t, a.smoke)
    else:
        full(a.root, a.xml_dir, a.titles, a.target_chars, a.min_matches)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
