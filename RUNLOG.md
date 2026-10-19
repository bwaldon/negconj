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
