# Work plan: extracting negative conjunctions from the U.S. Code

**For:** a Claude Code agent running locally on the user's laptop, running the heavy stages on Modal
**Goal:** find every sentence in the primary statutory text of the U.S. Code matching `… not … and …`, constituency-parse all of them with Stanza (single best tree per sentence), return every parse, and flag the sentences where *not* is the child of a VP that dominates the conjunctive coordination.

This adapts the method in the Champollion et al. amicus brief (*Campos-Chaves v. Garland*, Nos. 22-674 & 22-884), which searched a U.S. Code corpus for negative **disjunctions** using Stanza and NLTK and then reviewed hits by hand, to negative **conjunctions**. Text extraction is done by the provided module `uslm_extract.py`, which implements the procedure in `Another_copy_of_extract.ipynb` with the fixes listed in §2.

---

## 0. Ground rules for the agent

- **Stop and ask the user before:** (a) starting the full run if the pilot's cost projection (§4) exceeds the remaining Modal credit or a cap the user sets; (b) deleting the Modal Volume or anything on it; (c) changing any of the scientific definitions in §5 or §6.
- The user has a Modal account. Use the token already configured on the laptop (`modal profile current` should succeed); never print it.
- Do all development and unit testing **locally on Titles 29 and 18** before running anything on Modal. Modal should only ever run code that already passes tests locally.
- Every remote step must be **idempotent**: GPU functions on Modal are always preemptible (non-preemptible mode isn't available for GPU functions), and a preempted call is restarted with the same input.
- Commit to a local git repo at the end of each phase. Every output file is reproducible from a commit hash plus the run manifest (§8).
- Keep a running `RUNLOG.md` of decisions, counts, and anomalies; the user will read this, not the terminal history.

## 1. Repository layout (create locally)

```
negconj/
  pyproject.toml            # pinned deps: modal, beautifulsoup4, lxml, pandas, pyarrow, stanza==<pinned>, nltk, tqdm, pytest
  negconj/
    uslm_extract.py         # PROVIDED: notes stripping, unit extraction, identifiers, chapeau condensation
    download.py             # fetch + unzip USLM release point
    shards.py               # group sections into work shards; progress bookkeeping
    segment.py              # sentence segmentation + regex filter
    parse.py                # Stanza constituency parsing (batched, OOM-safe)
    classify.py             # tree predicates + C2 classification
    report.py               # summary tables, review sample, manifest, checksums
    modal_app.py            # Modal image, Volume, functions, orchestrator (§4-§5)
  scripts/
    launch.py               # deploys the app and spawns the orchestrator
    pull_results.sh         # modal volume get + local checksum verification
  tests/
    test_uslm_extract.py    # PROVIDED: fixture tests for every fix in §2
    test_shards.py  test_segment.py  test_classify.py  test_regression.py
  RUNLOG.md
```

`uslm_extract.py` and `test_uslm_extract.py` are provided by the user. Use them as-is; if a real title exposes a bug, fix it in the module, add a fixture test reproducing it, and log the change in `RUNLOG.md`.

## 2. Extraction: what `uslm_extract.py` does

`extract_title(path) -> (elements, units)` parses one USLM title file and returns two DataFrames. Fixes relative to the notebook (each has a test in `test_uslm_extract.py`):

1. **Text joins.** Spaces are inserted only at block boundaries (`p`, `content`, level elements, `num`, …), never at inline tags (`ref`, `i`, …). The notebook's bare `get_text()` glued adjacent `<p>` blocks (`prosperity.The`); `get_text(" ")` would split words around inline tags. Unicode spaces (U+202F narrow no-break space, U+00A0 no-break space, …) are normalized.
2. **Footnote markers.** `<ref class="footnoteRef">` is removed along with `<note>`/`<notes>` (the stray `1` in the notebook's `section 134(d); 1`).
3. **Positions.** `parent_position` is the element's index among the Tag descendants of its `@id`-bearing anchor (anchor = 0), found by object identity. The notebook's `list(parent.children).index(el)` raised on non-direct children, collided on textually identical siblings, and crashed when the element carried its own `@id`. Positions differ numerically from the notebook's, since whitespace strings no longer count. Uniqueness of `(parent_id, parent_position)` is a hard assertion per title.
4. **`@identifier` and `section`.** Every row records the nearest `@identifier` (e.g. `/us/usc/t29/s2/a/1`), a citation path that is stable across release points, and the `@identifier` of its enclosing `<section>`, which is the checkpoint key for §5.
5. **Structural chapeau condensation.** A chapeau-bearing level is rendered by walking its *direct* children in order (chapeau, sub-levels recursively, continuations, provisos), instead of the notebook's recursive indent-class search, which could pull in items or continuations from nested structures.
6. **Two strings per unit.** `text_display` keeps sub-level numbers (`(A)`, `(ii)`); `text_parse` drops them. Sub-level headings are dropped from both.
7. **No double counting.** Only the outermost chapeau-bearing level in a subtree becomes a condensed unit; elements inside it carry `consumed_by` and are not emitted again. Text inside `<quotedContent>` stays part of its enclosing element.
8. **Performance.** Rows are collected in lists; no per-row `pd.concat`.

Columns:
- `units`: `unit_id, title, is_appendix, section, identifier, parent_id, parent_position, elem_source (content|chapeau_condensed|continuation|proviso), text_display, text_parse, n_elements`
- `elements`: `elem_source, section, parent_id, parent_position, identifier, text, consumed_by`

Units outside any `<section>` have `section = None`; downstream code keys them as `{title}/_nosection` so they are still processed and checkpointed.

## 3. Phase 1: local build and tests (no cloud yet)

1. Scaffold the repo, pin dependencies. Pin the latest Stanza release at build time (1.11.1 as of Feb 2026; verify on PyPI) and record the English constituency model package Stanza reports when the pipeline loads.
2. Download Titles 29 and 18:
   `https://uscode.house.gov/download/releasepoints/us/pl/119/121/xml_usc29@119-121.zip`
   `https://uscode.house.gov/download/releasepoints/us/pl/119/121/xml_usc18@119-121.zip`
3. Run `pytest tests/test_uslm_extract.py`, then `python -I tests/test_uslm_extract.py usc29.xml usc18.xml`, which runs `check_real_titles` on both. Log to `RUNLOG.md`:
   - element counts per type for Title 29 next to the notebook's (11,240 content / 2,557 chapeau / 323 continuation / 1 proviso at release point 118-106; expect drift at 119-121);
   - the condensed text of the notebook's example chapeau (the workforce and labor market information system provision, `chapeaus[10]` in the notebook), checking that (A)–(H) and their nested items appear exactly once, in order;
   - a spot-check for glued words (`[a-z][.;][A-Z]` with no space) and any surviving footnote digits;
   - how many units have `section = None`, and in which titles.
4. **Regression fixture: Pulsifer.** Assert that 18 U.S.C. § 3553(f) (the safety-valve provision at issue in *Pulsifer v. United States*: "does not have— (A)…; (B)…; and (C)…") comes out as **one condensed unit** containing the negation in the chapeau and the `and` before (C). It must survive every stage through to the final classified table.
5. Implement `shards.py`, `segment.py`, `parse.py` and `classify.py` with unit tests: hand-written trees for §6; and for checkpointing, a test that kills shard processing partway (raise after N sections), reruns it, and checks the merged output equals an uninterrupted run (no missing and no duplicate `sent_id`s).
6. Run a CPU-only Stanza smoke test on ~50 Title 29 matches to confirm the end-to-end shapes work.

**Checkpoint:** show the user the Title 29 counts at each stage, the Pulsifer tree with its C2 result, and 10 random flagged sentences with their trees, and get a go-ahead before using Modal.

## 4. Phase 2: Modal setup and pilot

**Modal objects** (all in `modal_app.py`):

- **Image:** Debian slim with Python 3.12, the pinned dependencies, and the `negconj` package. Download the Stanza English models (tokenize, mwt, pos, constituency) **at image build time** so containers start with them on disk.
- **Volume:** `negconj-data`, mounted at `/data`. Prefer a v2 Volume, which handles many concurrent writers to distinct files; if only v1 is available, cap concurrent writers at 5 (Modal's guidance for v1). Each file has exactly one writer at a time; call `vol.commit()` after each checkpoint.
- **Functions:**

| Function | Hardware | Preemptible? | Work unit |
|---|---|---|---|
| `download` | CPU | non-preemptible | release point |
| `extract_title` | CPU, high memory (start at 32 GiB; raise if the pilot shows it's needed) | non-preemptible | one title |
| `plan_shards` | CPU | non-preemptible | whole corpus |
| `process_shard` | 1 x L4 GPU, `timeout` about 2 h, `retries=3` | preemptible (always, for GPU) | one shard of sections, checkpointed per section |
| `merge`, `classify`, `report` | CPU | non-preemptible | whole corpus |
| `orchestrate` | CPU, `timeout` up to 24 h | non-preemptible | runs the stages in order |

Non-preemptible CPU functions cost 3x the list CPU and memory price, which is negligible at this scale.

**Launching.** Deploy the app and spawn `orchestrate` remotely (`scripts/launch.py`), so the run doesn't depend on the laptop staying connected. The orchestrator skips anything already done, so relaunching after any failure resumes rather than restarts. Follow progress with Modal's app logs and the progress files on the Volume.

**Pilot.**
1. Run the full pipeline on Titles 29 and 18 only.
2. Separately, run `extract_title` alone on the largest titles (26 and 42) to measure peak memory and runtime; set the `extract_title` memory request from this.
3. From Modal's usage figures for the pilot, compute GPU-seconds per matched sentence and the number of prefilter survivors per title, and project the cost of the full run. Log the projection in `RUNLOG.md`.
4. Compare pilot outputs with the local Phase 1 outputs for Titles 29 and 18: identical units and matches; parses should match too, except for any GPU vs. CPU numeric differences, which should be logged.

**Checkpoint:** show the user the projection and the pilot's Pulsifer result. Proceed only with approval if the projection exceeds the remaining Starter credit ($30/month) or a cap the user has set.

## 5. Phase 3: full run on Modal

Stages, run in order by `orchestrate`. Each writes to the Volume and is skipped if its output already exists.

1. **Download** `https://uscode.house.gov/download/releasepoints/us/pl/119/121/xml_uscAll@119-121.zip` to `/data/raw/`. Before downloading, re-check `https://uscode.house.gov/download.html` for a newer release point and use whichever is current; record it in the manifest. (The OLRC site notes it is running in a temporary format, so if the download fails, stop and tell the user rather than switching to an unofficial mirror.) Record the zip's SHA-256. Appendix files (5a, 11a, 18a, 28a, 50a) are included; `uslm_extract` tags them `is_appendix=True`.
2. **Extract** each title with `uslm_extract.extract_title` (one `extract_title` call per title, in parallel) → `/data/units/{title}.parquet` and `/data/elements/{title}.parquet`. Write to a temporary name and rename on completion, so a half-written file is never mistaken for a finished one.
3. **Plan shards.** Apply the cheap prefilter (keep units whose lowercased `text_parse` contains both `\bnot\b` and `\band\b`), then group consecutive sections within each title into shards sized to finish well inside the GPU function's timeout (calibrate from the pilot, e.g. ~20 minutes of GPU work each). A large title becomes many shards; a small title may be one. Write the plan to `/data/shards.json` (shard ID → ordered list of section keys). The plan is deterministic, so rerunning it produces the same shards.
4. **Process shards** with `process_shard.map(...)`, at most 10 containers at once (the Starter plan's GPU concurrency limit; 5 if on a v1 Volume). The orchestrator launches shards in waves and sums the GPU time each reports; if the running total passes the approved budget, it stops launching new waves and logs where it stopped.

   Inside one shard:
   1. Load Stanza once per container.
   2. Read `/data/progress/{shard_id}.done` (section keys already finished) and skip those sections.
   3. Take the next sections in order until they contain at least B sentences to parse (e.g. 256). Never split a section across batches.
   4. **Sentence segmentation** of their units with Stanza `tokenize`. Legal text is hostile to sentence splitters (`U.S.C.`, `et seq.`, `No.`, `Pub. L.`); a post-pass re-merges splits right after those abbreviations, and the rate is logged. Each sentence gets ID `{unit_id}_{chunkno}`, following the notebook's `parent_id_position_chunkno` scheme.
   5. **Regex** on each sentence, case-insensitive: `\bnot\b.*\band\b` (`not` precedes `and`, whole words).
   6. **Parse every match** with Stanza `processors="tokenize,mwt,pos,constituency"`, `tokenize_pretokenized=True` (reuse the tokens from step 4 so segmentation doesn't change), on GPU. Stanza returns a single best tree per sentence; that tree is what's stored and classified. Sort by length within the batch. On GPU out-of-memory, retry the sentence alone, then on CPU; never silently drop one. A sentence that fails entirely goes to `/data/failures/{shard_id}.jsonl` with its ID.
   7. **Checkpoint:** write the batch's rows to `/data/parses/{shard_id}/part-{k:05d}.parquet` (columns: `sent_id, unit_id, section, title, text, tokens, xpos, n_tokens, tree, stanza_version, model_package`), then append the batch's section keys to `/data/progress/{shard_id}.done`, then `vol.commit()`. A section counts as done only once its rows are written.
   8. Return the shard's GPU seconds and counts to the orchestrator.

   If a container is preempted between writing a part file and recording its sections as done, the restarted call re-parses those sections and writes them again; `merge` drops the duplicates.

   File counts stay bounded: one progress file plus a handful of part files per shard, so hundreds to low thousands of files in total, not one file per section.
5. **Merge** all part files → `/data/results/parses.parquet` (one row per matched sentence; the matches table is the same rows without trees). Drop duplicate `sent_id`s, asserting the duplicates are identical. Assert every section in `shards.json` appears in its shard's `.done` file and that every prefilter-surviving unit was segmented.

## 6. Phase 4: structural classification (`classify.py`)

Runs as the `classify` CPU function over the merged parses; it can be rerun with a changed definition without reparsing.

Convert each Stanza tree to an `nltk.tree.ParentedTree` via `ParentedTree.fromstring(str(tree))`, so the predicates don't depend on Stanza's tree API.

**Definitions** (tell the user if any of these needs to change before the run, not after):

- **not-node**: a preterminal whose leaf lowercases to `not` (normally tagged `RB`). A sentence can have several.
- **coordination node**: for each preterminal `CC` whose leaf lowercases to `and`, the node immediately dominating it (in PTB flat coordination, the CC is a sister of the conjuncts under that node; it may be NP, VP, S, ADJP, PP, UCP, …). Record its label and number of conjuncts.
- **dominates(a, b)**: a is a proper ancestor of b.

```python
from nltk.tree import ParentedTree

def dominates(a, b):
    p = b.parent()
    while p is not None:
        if p is a:
            return True
        p = p.parent()
    return False

def c2(not_node, coord_node, relaxed=False):
    p = not_node.parent()
    if relaxed:                      # skip unary projections such as (ADVP (RB not))
        while p is not None and len(p) == 1:
            p = p.parent()
    return p is not None and p.label().startswith("VP") and dominates(p, coord_node)
```

**Criterion C2**, computed for every (not-node, coordination-node) pair in the sentence: *not*'s parent is labelled `VP`, and that VP dominates the coordination node.
- *strict*: the literal parent must be VP; this is the flagging criterion.
- *relaxed*: first skip unary projections above *not* (e.g. `(ADVP (RB not))`), then require VP. Reported as a separate column because Stanza sometimes wraps *not* in a unary ADVP.

Sentence-level outputs: `flagged = any(C2 strict)`, `flagged_relaxed = any(C2 relaxed)`. Per-pair columns: `not_parent_label`, `coord_label`, `n_conjuncts`, `tree_distance`.

**Unit tests on hand-built trees** (expected values verified against the definitions above; all must pass before parsing):

| Tree (schematic) | C2 strict | C2 relaxed |
|---|---|---|
| `(S (NP X) (VP (MD shall) (RB not) (VP (VB have) (NP (NP b) (CC and) (NP c)))))` (Pulsifer shape) | T | T |
| `(S (NP X) (VP (MD shall) (RB not) (VP (VP (VB sell)) (CC and) (VP (VB buy)))))` | T | T |
| `(S (NP X) (VP (MD shall) (ADVP (RB not)) (VP (VB have) (NP (NP b) (CC and) (NP c)))))` | F | T |
| `(S (NP X) (VP (MD shall) (RB not) (VP (VB act))) (PP (IN except) (PP (IN in) (NP (NP A) (CC and) (NP B)))))` | F | F |
| `(S (S (NP X) (VP (VBZ is) (RB not) (ADJP red))) (CC and) (S (NP Y) (VP (VBZ is) (ADJP blue))))` | F | F |

Outputs (in `/data/results/`):
- `pairs.parquet` — one row per (not, and) pair.
- `sentences_classified.parquet` — one row per matched sentence, **all** matched sentences, with their parse and C2 flags (this is the "all parses" deliverable).
- `flagged.parquet` / `flagged.csv` — subset where `flagged` is true, with `section`, `identifier`, `text_display`, and tree.

## 7. Phase 5: verification, return, cleanup

1. **Regression:** 18 U.S.C. § 3553(f) (Pulsifer) appears in `parses.parquet` and in `sentences_classified.parquet`; report its tree and C2 result explicitly in `RUNLOG.md`, whether or not it's flagged.
2. **Consistency checks** (in `report`): every match has exactly one parse row or a failure record; counts reconcile across stages; no duplicate `sent_id`s; no sentence appears both standalone and inside a condensed chapeau; every planned section is marked done.
3. **Manual-review sample** (the brief's authors hand-reviewed their hits): `review_sample.csv` with a stratified random sample — e.g. 100 flagged and 100 unflagged matches, stratified by title and by `elem_source` — with blank columns `reviewer_is_negconj`, `reviewer_scope_reading`, `notes`. Use a fixed seed.
4. **Pull back:** `report` writes `/data/results/CHECKSUMS.sha256`. `scripts/pull_results.sh` runs `modal volume get negconj-data results ./results` and verifies every checksum locally.
5. **Cleanup:** nothing is billed while no function is running, so there's no instance to tear down. The Volume's storage cost is small; **ask the user** before deleting it (`modal volume delete`), since it holds the raw XML, extracted units and checkpoints needed to rerun any stage.

## 8. Final deliverables on the laptop

```
results/
  RUN_MANIFEST.json            # release point, zip SHA-256, git commit, stanza version, model package,
                               # GPU type, GPU-seconds, wall time, and counts at every stage
  CHECKSUMS.sha256
  units/*.parquet              # all extracted primary-text units, per title
  elements/*.parquet           # element-level table, per title (traceability)
  parses.parquet               # every regex match with its tree
  pairs.parquet                # per (not, and) pair C2 results
  sentences_classified.parquet # every matched sentence + parse + C2 flags
  flagged.csv                  # the target set
  review_sample.csv            # for manual review
  summary.md                   # stage counts; flagged counts by title and coordination label;
                               # the Pulsifer case; failures; known limitations
RUNLOG.md
```

`summary.md` should state the known limitations plainly: regex recall is bounded by the `not … and` surface pattern (no `n't`, `cannot`, `nor`, `without`, `no`); parser errors on very long condensed chapeau sentences; the flagging criterion is syntactic (the negated VP dominating the coordination), not a judgment about which scope reading is correct in context, which, as the brief stresses, depends on context.
