# RUNLOG

Running log of decisions, counts and anomalies for the negative-conjunction study
(see `negconj_workplan.md`). Newest phase last.

## Phase 1 — local build and tests (2026-10-19)

### Environment and pins
- Python 3.12 (Homebrew) in `.venv`; macOS arm64, 14 cores, 24 GB.
- Pinned (latest on PyPI at build time): stanza **1.15.0** (the plan's note said 1.11.1 as of Feb 2026;
  1.15.0 is current), modal 1.6.2, beautifulsoup4 4.15.0, lxml 6.1.3, pandas 3.0.6, pyarrow 26.0.0,
  nltk 3.10.3, tqdm 4.70.1, pytest 9.1.1 (torch 2.14.1 came in via stanza).
- Stanza English models reported when the pipeline loads:
  `tokenize=combined_nocharlm; mwt=combined; pos=combined_charlm; constituency=ptb3-revised_charlm`
  (pretrain `conll17`, charlm `1billion`). Stored per row as `model_package`.
- Modal: `modal profile current` → `default` (token already configured; not printed).

### Layout deviations from §1
- `uslm_extract.py` lives in `negconj/` and its test in `tests/`; the test's `sys.path` line now
  points at `../negconj` (the only change to how it imports).
- Added `negconj/stages.py` (file-level stage wiring shared by the local runner and Modal) and
  `scripts/run_local.py` (local CPU run in the same on-disk layout as the Volume, under `data/local/`).

### Downloads (release point 119-121)
| file | SHA-256 |
|---|---|
| `xml_usc29@119-121.zip` | `359198f6ec9a40212485a0b111f93536ce16a0063878367c39e8ca2d083b622f` |
| `xml_usc18@119-121.zip` | `81e57f161a5429473af7c1bf3f841aedf9b9f00782223adeda9dad892d16e1ca` |

### Extraction checks (§3.3)
- `pytest tests/test_uslm_extract.py`: all provided tests pass; `python -I tests/test_uslm_extract.py usc29.xml usc18.xml`
  runs `check_real_titles` cleanly.
- Element counts, Title 29 (119-121 vs. the notebook at 118-106):

  | | content | chapeau | continuation | proviso |
  |---|---|---|---|---|
  | notebook 118-106 | 11,240 | 2,557 | 323 | 1 |
  | this run 119-121 | 11,366 | 2,592 | 324 | 1 |

  Drift is small and in the expected direction (new law). Title 18: 8,669 / 1,843 / 332 / 2.
- Units: Title 29 4,940 (3,342 content, 1,591 condensed chapeau, 7 continuation); Title 18 4,217
  (3,008 / 1,207 / 2).
- Notebook example chapeau (`chapeaus[10]`): 29 U.S.C. 49l–2(a)(1), the workforce and labor market
  information system provision. Its condensed unit (23 elements) contains (A)–(H) and every nested
  (i)–(iv) item exactly once and in order; "section 134(d);" carries no stray footnote digit.
- Glued words (`[a-z][.;][A-Z]`): 0 units in either title.
- Units with `section = None`: 0 in Title 29, 0 in Title 18.

### Bug fixed in `uslm_extract.py`: repeated-footnote `<sup>` markers
- At 119-121, a footnote cited a second time is marked with a bare `<sup>1</sup>` and no
  `<ref class="footnoteRef">`, so the digit survived: e.g. 29 U.S.C. 218c(a)(2) rendered as
  "…amendment made by this title); 1". 47 `<sup>` in Title 29 and 66 in Title 18, all digits.
- One `<sup>` is a real exponent: 18 U.S.C. 841, "10−<sup>4</sup> Pa".
- Fix: `strip_non_primary` also removes a digit-only `<sup>` unless the text right before it ends in a
  digit or minus sign. Fixture test: `test_repeated_footnote_sup_removed_exponent_kept`. Remaining
  "`; 1`"-style hits in both titles were checked by hand and are real numbers ("8 hours", "88 Stat. 69").

### Pulsifer regression fixture (§3.4)
- 18 U.S.C. 3553(f) comes out as **one** `chapeau_condensed` unit (10 elements), unit_id
  `idd424bda2-b5bd-11f1-a405-86f9ef1fd617_0_chap`. Segmentation gives 2 sentences; sentence `_0`
  (365 tokens) holds "the defendant does not have— …; and a prior 2-point violent offense" and
  matches the regex. `tests/test_regression.py` checks extraction, segmentation and (when present)
  the parse and classified tables.

### Segmentation (§5.4.4)
- Stanza's tokenizer already keeps "U.S.C.", "No.", "Pub. L." intact. A survey of split points
  in Title 29 showed the harmful splits are elsewhere: after "—" (the condensed-chapeau join,
  e.g. "this chapter— A case shall…"), after "," (": Provided, That…") and after ":".
- Re-merge rules (`segment.merge_reason`): left side lacks . ? ! (`nonfinal`); right side starts
  lowercase (`lower`) or with closing punctuation (`punct`); left side ends in a never-final
  abbreviation (`abbrev`: No., Pub., L., U.S.C., Stat., months, …). "et seq." can end a sentence,
  so it merges only under `lower`. **Going beyond the plan's abbreviation list is a judgment call;
  flagged for the user.**
- Titles 29+18: 5,853 raw splits → 352 re-merged (6.0%; 327 nonfinal, 24 lower, 1 abbrev) → 5,501 sentences.
- **Bug found and fixed:** Stanza's English tokenizer sometimes emits whitespace-containing tokens
  ("United States", "180 days", "appropriate official"; 17 in Titles 29+18). In a bracketed tree a
  leaf with a space reads back as two leaves. The parser's leaf-count check caught every case, so
  14 sentences went to failures in the first local run rather than being mis-parsed.
  `segment.split_spaced` now splits such tokens at the whitespace, keeping offsets (test:
  `test_split_spaced_tokens`). The first run was moved to `data/local_v1/` and everything was rerun.

### Parser
- Stanza `LSTMModel`, in-order transition-based (shift-reduce family), charlm features, no
  transformer (`bert_model=None`); package `ptb3-revised_charlm`.
- Local Apple GPU (MPS) is *slower* than CPU for this parser (40 sentences: 28.0 s MPS vs 17.6 s
  CPU) and gives identical trees (40/40), so Phase 1 ran on CPU.
- The user was told that Stanza's transformer-backed English constituency models exist (more accurate, slower);
  keeping the default unless the user decides otherwise.

### Full local CPU run, Titles 29 + 18 (`data/local/`, `scripts/run_local.py`)
| | T18 | T29 |
|---|---|---|
| units | 4,217 | 4,940 |
| prefilter units / sections | 1,255 / 695 | 1,167 / 375 |
| shards (target 400k chars) | 4 | 4 |
| raw splits → sentences | 2,591 → 2,444 | 3,262 → 3,057 |
| regex matches | 726 | 890 |
| parsed / failed | 726 / 0 | 890 / 0 |
| (not, and) pairs | 5,258 | 6,797 |
| flagged (strict) / relaxed | 180 / 181 | 324 / 326 |

- Total parse wall time on CPU: 661 s for 1,616 sentences (≈0.41 s/sentence). Match length in
  tokens: median 103, p90 301, p99 934, max 2,204.
- By source: condensed chapeau 302 / 963 flagged; content 202 / 652; continuation 0 / 1.
- `report_stage` consistency checks: all pass (one parse or failure per match, no duplicate
  sent_ids, classified = parses, all planned sections done, no standalone/condensed overlap).

### Pulsifer result (local CPU parse)
- Sentence `…_0_chap_0` is **flagged = True**, but **not because of the Pulsifer pair**.
  Its 3 strict pairs all come from paragraph (4) ("was **not** an organizer … **and** was not engaged").
- The scope-relevant pair ("does **not** have" × "; **and** a prior 2-point violent offense") is
  C2 strict **False**, relaxed **False**. The parser builds
  `(S (S the defendant does not have — (A)) (: ;) (NP (B)) (: ;) (CC and) (NP (C)))`: the
  coordination node is an S *above* the negated VP, which covers only (A).
- Takeaway: a sentence-level flag on a long condensed sentence can come from an unrelated
  (not, and) pair; `pairs.parquet` holds the per-pair answer.

### Observations from 10 random flagged sentences (seed 0)
Clean hits: 29/1426(g) "shall not apply … in [critical and declining] status"; 29/791(e);
18/3509(d)(4) "[the welfare and well-being]"; 29/1191a(c)(1); 18/845(a).
False-positive patterns:
1. **Parser overreach on long chapeaus.** In 18/3626(g) the negated VP swallows the following
   definitions.
2. **"not later than"/"not less than" read as VP negation.** 18/3174(d)(2) "shall not later than
   ten days … transmit [to the Congress and to the Judicial Conference]".
3. **"between X and Y" ranges and coordinations inside relative clauses/modifiers.** These count because
   C2 asks only for dominance (18/5037(d)(2) "between [18 and 21] years"; 29/1461(b)).
Also a mis-coordination: 29/214(d).

### Decision: switch to the transformer parser (user, 2026-10-19)
- The user chose Stanza's transformer-backed parser. Configuration (Stanza's `default_accurate`
  pairing for these two processors): pos `combined_electra-large`, constituency
  `ptb3-revised_electra-large`, transformer `google/electra-large-discriminator`. Tokenizer
  unchanged (`combined_nocharlm`), so segmentation and sent_ids are unchanged.
  `parse.PARSER_CONFIGS["charlm"]` keeps the old setup selectable. Added pin `transformers==5.20.2`.
- Long inputs: ELECTRA's limit is 512 subwords. Stanza's `build_cloned_features` handles longer
  sentences with overlapping 512-subword windows (20-subword left context) and stitches the
  embeddings. Sentences past ~500 subwords therefore get contextual embeddings from local windows
  only. The longest match (2,204 tokens) parses without error (98 s on MPS).
- Speed (40-sentence sample, 5,245 tokens): CPU 64.9 s (81 tok/s), about 4.5× slower than charlm.
  MPS 69.3 s, no faster. CPU and MPS trees identical (40/40).
- Trees differ from charlm on 40/40 sample sentences; the sentence flag changes on 3/40 (15 → 16).
- **Pulsifer under ELECTRA:** the key pair ("does not have" × "; and (C)") is **C2 strict True,
  relaxed True**. Caveat: the coordination node is one flat 8-member NP under "have" containing
  (A), (B), (C) *and* paragraphs (2)–(5). Right for (A)–(C), but the bracketing is too wide.
- The 10 charlm-flagged sentences under ELECTRA: 3 false positives become unflagged (18/3626(g)
  chapeau overreach, 18/3174(d)(2) "not later than", 29/1461(b) relative clause). 5037(d)(2)
  "between 18 and 21" and 29/214(d) mis-coordination remain flagged. The 5 clean hits stay flagged.
- Modal image now also bakes in the ELECTRA Stanza models and HF weights (`HF_HOME=/opt/hf`,
  `HF_HUB_OFFLINE=1` at runtime).
- charlm local results kept in `data/local_charlm/`; the local reference is being rerun with ELECTRA.
- User decision: skip the local ELECTRA reference run (stopped partway; partial output in
  `data/local_electra_partial/`) and go to Modal. The pilot comparison (§4.4) can therefore check
  units, sentences and regex matches against the local charlm run (segmentation is identical),
  but not trees.

## Phase 2 — Modal setup and pilot (2026-10-19)

### Checkpoint granularity (user-approved change)
- Pilot finding: with `min_matches=256`, a pilot shard (~200–250 matches) checkpointed only at its
  end, so two preemptions in the pilot discarded whole shards of parsing.
- Default `min_matches` is now **32** (run_shard, Modal functions, launch.py, run_local.py):
  a preemption loses at most one small batch. Takes effect at the next deploy (after the pilot).

### Pilot run (Titles 29 + 18 on Modal; orchestrator call fc-01M5AZTZM6WTB544CZN83KZ1BB, commit 8b1d282)
- Deploy: needed the Modal token from `.env` (`scripts/modal_env.sh` exports it without echoing;
  `modal profile current` named a profile with no stored token). Slip: `modal token info` printed
  the token *ID* (not the secret) once in tool output.
- Image built in ~265 s (CUDA torch, Stanza charlm + ELECTRA models, HF ELECTRA weights).
- OLRC page moved: `/download/download.html` → `/download.html`. The pilot logged "site current =
  None"; fixed in 151a7e2 (detects 119-121; the stale 404-page banner said 119-120).
- 2 GPU preemptions. With `min_matches=256` (pre-86fb756) those attempts lost whole shards; the
  retries completed. Merge dropped 0 duplicate rows.
- Results: extraction 9,157 units; plan 8 shards / 1,070 sections; 5,853 raw splits → 352 merged →
  5,501 sentences; 1,616 matches; **1,616 parsed, 0 failures**; 12,055 pairs; **flagged 487**,
  relaxed 488 (ELECTRA; local charlm gave 504 / 507). Report checks all OK; 14 checksums verified
  after `scripts/pull_results.sh pilot` (script fixed: `modal volume get` with a destination path
  created an empty file and failed).
- Comparison with local Phase 1: units (both titles), segments, match sent_ids and tokens
  **identical**. Trees: local reference was charlm, so instead L4 vs local-CPU ELECTRA was compared
  on the 51 sentences parsed locally with ELECTRA (40-sentence sample, 10 flagged, Pulsifer):
  **51/51 identical**. No GPU/CPU numeric differences observed.
- **Pulsifer (pilot, ELECTRA on L4): flagged True / relaxed True; key pair ("does not have" ×
  "; and (C)") C2 strict True, relaxed True**, coord NP with 8 members (same caveat as above).
- Extract sizing (`measure_extract`, Titles 26 & 42): T26 17,365 units, 22 s, peak 1.3 GiB;
  T42 48,428 units, 51 s, peak 2.4 GiB. `extract_title` memory lowered 32 → 8 GiB.
- GPU time per shard (s): 18-000 220, 18-001 257, 18-002 174, 18-003 10 (warm), 29-000 131,
  29-001 86 (warm), 29-002 180, 29-003 114; total 1,172 s reported for 2.68 M prefilter chars,
  255,123 matched tokens.
- Billing (Modal, app negconj, 2026-10-19, pilot + measurement + deploy): L4 $0.346
  (≈1,557 L4-s billed vs 1,172 s reported by completed calls: startup + preempted attempts),
  memory $0.203, CPU $0.142 → **≈ $0.69**. Month metered $1.02, all covered by credits.

### Full-corpus extraction (orchestrate run=full, stop_after=extract; call fc-01M5B0GCCB8FH127M9EHHZWB2G, commit f83b6b9)
- Release point **119-121** (OLRC download page reports 119-121 as current).
  `xml_uscAll@119-121.zip` SHA-256 recorded in `/data/raw/release_119-121_All.json` and the run manifest.
- 56 title files; 266,232 units; prefilter: 60,809 units, 25,054 sections, **60,843,487 chars**.
- Units with `section = None`: 2,558, all in appendices: 11a (961; prefilter 154 units, 88,820 chars),
  18a (443), 28a (1,154; prefilter 233 units, 179,596 chars). These are court rules, not sections.
  Each title's no-section units form one `{title}/_nosection` key and so one indivisible batch.
- Largest by prefilter chars: 42 (12.0 M), 26 (5.0 M), 15 (4.0 M), 16 (3.6 M), 12 (3.0 M), 07 (2.9 M).
- Title labels follow the zip's file names (`usc07.xml` → "07").

### Cost projection (§4.3)
- Pilot: 1,172 reported GPU-s / 2.68 M prefilter chars = 4.4e-4 s/char (includes model loading on
  cold containers); billed L4 time was 1.33× reported.
- Full corpus: 60.84 M chars → ≈ 26,600 reported GPU-s → ≈ 35,400 billed L4-s ≈ **$7.9 L4** + CPU/memory
  ≈ **$9–11 total**. Under the remaining Starter credit (~$29), and the user set no lower cap.
- Launch settings: `target_chars=2,000,000` (≈ 15 min GPU per shard), GPU budget stop at
  **45,000 reported GPU-s** (≈1.7× projection; worst case ≈ $13 L4).

## Phase 3 — full run (orchestrator call fc-01M5B0KW1HNH3Y0K6MVP2NV1TT, commit 4e53ed0)
- 72 shards planned (target 2 M chars). 69 completed with 0 parse failures: 41,559 trees,
  29,583 reported GPU-s. Cost at 22:05 UTC: L4 $5.84, memory $1.63, CPU $1.09 (billing lags).
- **Stuck:** shards 34-000, 42-001, 42-002 stopped progressing (no new sections 22:45 → 22:54 UTC).
  Logs show CUDA OOM on sentences up to ~21k subwords, then the CPU fallback exceeding the
  container's memory ("Runner was terminated while exceeding its memory request"). The container
  restarts and re-hits the same batch. The GPU budget stop counts only completed shards, so it cannot
  catch this loop.
- My `modal app stop negconj` was blocked by the permission system; the user was asked to stop the app.
- Local scan of the stuck shards' remaining sections: 2,150 matches; longest matched sentences
  42 U.S.C. 1396a (19,468 tokens), 34 U.S.C. 11133 (6,109), 42 U.S.C. 671 (6,104),
  42 U.S.C. 1395l (5,215), 42 U.S.C. 1396d (2,767); every other one ≤ 2,451.
  Proposed (awaiting the user): a token cap (e.g. 2,500) with over-long sentences recorded as
  failures; no CPU retry for oversized inputs; budget stop that also counts in-flight shards.

### Partial results pulled (`results_full_partial/`)
- Merged locally from the 69 completed shards (Titles 34 and 42 only partly covered).
  41,559 matches = 41,559 parses, 0 failures, 0 duplicates; 278,434 pairs; flagged 12,425,
  relaxed 12,444. Consistency checks OK (for the completed shards); local checksums verified.
- Pulsifer: flagged True / relaxed True (same as pilot).
- Flagged sentences with a strict pair inside an if-clause: 972.
- Dashboard: `results_full_partial/dashboard.html` (flagged only, 41.8 MB).
- 2026-10-20: Modal app no longer running (0 containers; app not listed as deployed). Spending stopped.

## Publication (2026-10-20)
- Repo pushed to <https://github.com/bwaldon/negconj> (public; required for GitHub Pages on this account).
  Before pushing, the history was scanned for the `.env` token values and token-like strings: none.
  `.env`, `data/` and `results*/` were never committed.
- Dashboard published at <https://bwaldon.github.io/negconj/> from an orphan `gh-pages` branch
  (single `index.html`, 41.8 MB): flagged sentences from the 69 finished shards.

## Phase 4 — finish the conjunction run; negative-disjunction run (2026-10-21)

### Length cap for the stuck shards (user decision)
- Local CPU measurements (ELECTRA, 42 U.S.C. 1395l chapeau sentence, first N tokens):
  peak RSS above the loaded pipeline +2.4 GiB at 1,000 tokens, +7.3 GiB at 2,000; the 4,000-token
  prefix and the full 5,215-token sentence were killed on the 24 GB laptop. 42 U.S.C. 1396d's
  2,767-token sentence: 54 s, +4.1 GiB. Growth is roughly quadratic, so the 19,468-token sentence
  would need hundreds of GiB; the parser itself has no attention layers (`pattn_num_layers=0`,
  `use_lattn=False`) and runs under `no_grad`; the profile is dominated by `addmm`.
- **Decision (user): sentences above a length cap are not parsed.** Cap = **4,000 tokens**
  (`modal_app.MAX_GPU_TOKENS`): the full run parsed every match up to 3,857 tokens on L4s; every
  sentence that stalled it was ≥ 5,215 tokens. Over-cap matches are written to
  `failures/{shard}.jsonl` with error `too_long: N tokens > max_tokens 4000`, so they are counted
  (`parse_failures_too_long`) and never silently dropped. Applies to both runs.

### Negative disjunction (`conn="or"`), mirroring the conjunction code
- The user asked that the disjunction run mirror the *code* used for `not … and` (not the
  work plan). Only the connective changes: prefilter `\bnot\b` & `\bor\b`; sentence regex
  `\bnot\b.*\bor\b` (case-insensitive, DOTALL); coordination node = parent of a `CC` whose word
  is `or`; C2 strict / relaxed unchanged. Same extraction, segmentation and re-merge rules, same
  ELECTRA parser, same checkpointing. Column names follow the connective (`or_index`,
  `not_before_or`, `n_or_cc`). `conn="and"` (the default) is byte-for-byte the old behaviour.
- Regression case: *Campos-Chaves*, 8 U.S.C. § 1229a(b)(5)(C) (one condensed-chapeau unit:
  "… the alien did not receive notice in accordance with paragraph (1) or (2) …").
- Run `disj` on the Volume (`/data/disj`), its own extraction and shard plan.
- **Tree reuse.** Parsing does not depend on the connective, and 52.5 M of the 58.6 M `not … or`
  prefilter chars are in units that also pass the `not … and` prefilter. With `reuse_run="full"`,
  a disjunction shard takes the tree and xpos from the finished conjunction run's
  `results/parses.parquet` when the sent_id, the token list and the model package are all
  identical; anything else is parsed. Counted per shard (`reused`); the reused file's SHA-256 is
  in the disjunction manifest.

### Conjunction run finished (orchestrator call fc-01M5FS9MKC3GERB2RWNXDR86HT, commit d851771)
- The 3 stalled shards resumed after ~13 min waiting for L4 capacity: 34-000 (451 matches, 343 s),
  42-002 (705, 713 s), 42-001 (994, 997 s). Total reported GPU 31,635 s.
- Merge: 144,017 sentences (153,418 raw splits, 9,401 re-merged); **44,517 matches, 44,513 parsed,
  4 not parsed (too long)**: 42 U.S.C. 1396a (19,468 tokens), 34 U.S.C. 11133 (6,109),
  42 U.S.C. 671 (6,104), 42 U.S.C. 1395l (5,215). 0 duplicate rows. 308,155 pairs;
  **flagged 13,246, relaxed 13,265**. Report checks all OK; 122 checksums verified after
  `scripts/pull_results.sh` → `results/`.
- Pulsifer: flagged True / relaxed True (unchanged). The 41,559 sentences in the partial results are
  all present, with identical flags.
- Modal metered this month after the finish: $17.67 (credits cover it).
