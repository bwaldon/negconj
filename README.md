# negconj

Negative conjunctions (`… not … and …`) in the primary statutory text of the U.S. Code.
Every matching sentence is constituency-parsed with Stanza, and the pipeline flags those where *not*
is the child of a VP that dominates the conjunctive coordination (criterion C2). The method
adapts the negative-disjunction search in the Champollion et al. amicus brief in
*Campos-Chaves v. Garland* (Nos. 22-674 & 22-884).

**Dashboard:** <https://bwaldon.github.io/negconj/>. Flagged sentences with their trees,
filterable by title, coordination label and position (e.g. inside an *if*-clause).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Extract | `negconj/uslm_extract.py` | USLM XML → text units; notes and footnote markers removed; chapeaus condensed into one sentence |
| Plan | `negconj/shards.py` | prefilter units containing both *not* and *and*; group sections into GPU shards |
| Segment | `negconj/segment.py` | Stanza sentence splitting plus a re-merge pass for statutory punctuation; regex `\bnot\b.*\band\b` |
| Parse | `negconj/parse.py` | Stanza constituency parser (`ptb3-revised_electra-large`), one best tree per sentence |
| Classify | `negconj/classify.py` | C2 strict / relaxed for every (not, and) pair |
| Report | `negconj/report.py` | consistency checks, stratified review sample, summary, manifest, checksums |
| Run | `negconj/modal_app.py`, `scripts/launch.py` | Modal orchestration on L4 GPUs; resumable, checkpointed per section |

`negconj_workplan.md` is the full specification; `RUNLOG.md` records every decision, count and anomaly.

## Running

```sh
python3.12 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/pytest                                     # unit, checkpoint and Pulsifer regression tests
.venv/bin/python scripts/run_local.py --titles 29 18 # local CPU run
source scripts/modal_env.sh                          # Modal token from .env (not committed)
.venv/bin/python scripts/launch.py --run full --gpu-budget-seconds 45000
scripts/pull_results.sh                              # results + checksum verification
.venv/bin/python scripts/build_dashboard.py results --only-flagged
```

## Status

Release point 119-121. The full run finished 69 of 72 shards (41,559 parsed sentences, 0 parse
failures, 12,425 flagged). Three shards covering parts of Titles 34 and 42 stalled on sentences of
5,000–19,000 tokens and will be resumed once a length cap is set. The published dashboard is built
from the 69 finished shards.

## Limitations

Recall is bounded by the surface pattern (no *n't*, *cannot*, *nor*, *without*, *no*). Parser
accuracy degrades on very long condensed-chapeau sentences. The flag is syntactic: it does not
judge which scope reading is correct in context.
