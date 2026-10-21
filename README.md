# negconj

Negated connectives in the primary statutory text of the U.S. Code: negative conjunctions
(`… not … and …`) and negative disjunctions (`… not … or …`). Every matching sentence is
constituency-parsed with Stanza, and the pipeline flags those where *not* is the child of a VP that
dominates the coordination (criterion C2). The method adapts the negative-disjunction search in the
Champollion et al. amicus brief in *Campos-Chaves v. Garland* (Nos. 22-674 & 22-884).

**Dashboard:** <https://bwaldon.github.io/negconj/>. Flagged sentences from both runs, with their
trees. Pick a connective (*and*, *or*, or both) and search or filter by text, citation, title,
coordination label and position (e.g. inside an *if*-clause).

## Pipeline

A run studies one connective (`--conn and|or`). Only the prefilter, the sentence regex and the CC
word change between them; extraction, segmentation, parsing and C2 are shared code.

| Stage | Module | What it does |
|---|---|---|
| Extract | `negconj/uslm_extract.py` | USLM XML → text units; notes and footnote markers removed; chapeaus condensed into one sentence |
| Plan | `negconj/shards.py` | prefilter units containing both *not* and the connective; group sections into GPU shards |
| Segment | `negconj/segment.py` | Stanza sentence splitting plus a re-merge pass for statutory punctuation; regex `\bnot\b.*\b(and|or)\b` |
| Parse | `negconj/parse.py` | Stanza constituency parser (`ptb3-revised_electra-large`), one best tree per sentence; matches over 4,000 tokens are recorded as failures, not parsed |
| Classify | `negconj/classify.py` | C2 strict / relaxed for every (not, connective) pair |
| Report | `negconj/report.py` | consistency checks, stratified review sample, summary, manifest, checksums |
| Run | `negconj/modal_app.py`, `scripts/launch.py` | Modal orchestration on L4 GPUs; resumable, checkpointed per section |

The disjunction run reuses a tree from the conjunction run when a sentence matches both patterns
and its tokens and model package are identical (25,098 sentences).

`negconj_workplan.md` is the original specification for the conjunction study; `RUNLOG.md` records
every decision, count and anomaly, including where the code departs from the work plan.

## Running

```sh
python3.12 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/pytest                                     # unit, checkpoint and Pulsifer regression tests
.venv/bin/python scripts/run_local.py --titles 29 18 # local CPU run
source scripts/modal_env.sh                          # Modal token from .env (not committed)
.venv/bin/python scripts/launch.py --run full --target-chars 2000000 --gpu-budget-seconds 45000
.venv/bin/python scripts/launch.py --run disj --conn or --reuse-run full --target-chars 2000000 \
    --gpu-budget-seconds 30000 --no-deploy
scripts/watch_run.sh disj                            # follow a run's orchestrator log
scripts/pull_results.sh full                         # -> results/ (checksums verified)
scripts/pull_results.sh disj                         # -> results_disj/
.venv/bin/python scripts/build_dashboard.py --and results --or results_disj --out site
```

## Results (release point 119-121)

| | not … and | not … or |
|---|---|---|
| regex matches | 44,517 | 49,546 |
| parsed | 44,513 | 49,541 |
| not parsed (> 4,000 tokens) | 4 | 5 |
| flagged (C2 strict) | 13,246 | 28,623 |
| flagged (C2 relaxed) | 13,265 | 28,658 |
| regression case | Pulsifer, 18 U.S.C. § 3553(f): flagged | Campos-Chaves, 8 U.S.C. § 1229a(b)(5)(C): flagged |

Per-run detail is in `results*/summary.md` (not committed; produced by the report stage).

## Limitations

Recall is bounded by the surface pattern (no *n't*, *cannot*, *nor*, *without*, *no*). Parser
accuracy degrades on very long condensed-chapeau sentences, and the longest (over 4,000 tokens;
4 conjunction and 5 disjunction matches, in Titles 34 and 42) are not parsed at all. The flag is
syntactic: it does not judge which scope reading is correct in context.
