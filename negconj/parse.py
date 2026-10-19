"""Stanza constituency parsing of pretokenized sentences (work plan §5.4.6).

One best tree per sentence. Sentences are sorted by length and parsed in
sub-batches; on GPU out-of-memory (or any other error) the sub-batch is
retried one sentence at a time, then each failing sentence on a CPU pipeline.
A sentence that still fails is returned as a failure, never dropped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from nltk.tree import Tree

PROCESSORS = "tokenize,mwt,pos,constituency"

# Processor -> Stanza package. "electra" is Stanza's default_accurate pairing for
# pos + constituency; "charlm" is Stanza's default. The tokenizer is unused
# (input is pretokenized) and kept equal to segment.py's.
PARSER_CONFIGS = {
    "electra": {"tokenize": "combined_nocharlm", "mwt": "combined",
                "pos": "combined_electra-large", "constituency": "ptb3-revised_electra-large"},
    "charlm": {"tokenize": "combined_nocharlm", "mwt": "combined",
               "pos": "combined_charlm", "constituency": "ptb3-revised_charlm"},
}
DEFAULT_PARSER = "electra"


def build_pipeline(use_gpu: bool, config: str = DEFAULT_PARSER):
    import stanza
    return stanza.Pipeline("en", processors=PARSER_CONFIGS[config], tokenize_pretokenized=True,
                           use_gpu=use_gpu, logging_level="WARN", download_method=None)


def model_package(nlp) -> str:
    """e.g. 'tokenize=combined_nocharlm;mwt=combined;pos=combined_electra-large;
    constituency=ptb3-revised_electra-large;transformer=google/electra-large-discriminator'."""
    parts = []
    for name in PROCESSORS.split(","):
        proc = nlp.processors.get(name)
        if proc is not None:
            parts.append(f"{name}={Path(proc.config['model_path']).stem}")
    con = nlp.processors.get("constituency")
    bert = getattr(getattr(con, "_model", None), "args", {}).get("bert_model") if con else None
    if bert:
        parts.append(f"transformer={bert}")
    return ";".join(parts)


def _is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def _free_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class Parser:
    def __init__(self, use_gpu: bool, sub_batch_tokens: int = 6000, config: str = DEFAULT_PARSER):
        import stanza
        self.use_gpu = use_gpu
        self.config = config
        self.nlp = build_pipeline(use_gpu, config)
        self._cpu = None if use_gpu else self.nlp
        self.sub_batch_tokens = sub_batch_tokens
        self.stanza_version = stanza.__version__
        self.model_package = model_package(self.nlp)
        self.events: dict[str, int] = {"oom_batches": 0, "single_retries": 0, "cpu_fallbacks": 0}

    @property
    def cpu(self):
        if self._cpu is None:
            self._cpu = build_pipeline(False, self.config)
        return self._cpu

    @staticmethod
    def _run(nlp, token_lists: Sequence[Sequence[str]]) -> list[tuple[str, list[str]]]:
        doc = nlp([list(t) for t in token_lists])
        out = []
        for toks, s in zip(token_lists, doc.sentences):
            tree = "{}".format(s.constituency)
            n_leaves = len(Tree.fromstring(tree).leaves())
            if n_leaves != len(s.words):
                raise ValueError(f"tree has {n_leaves} leaves for {len(s.words)} words")
            out.append((tree, [w.xpos for w in s.words]))
        if len(out) != len(token_lists):
            raise ValueError("sentence count mismatch")
        return out

    def _one(self, toks) -> tuple[tuple[str, list[str]] | None, str | None]:
        self.events["single_retries"] += 1
        err = None
        if self.use_gpu:
            try:
                return self._run(self.nlp, [toks])[0], None
            except Exception as e:  # noqa: BLE001
                err = f"gpu: {type(e).__name__}: {e}"
                _free_gpu()
            self.events["cpu_fallbacks"] += 1
        try:
            return self._run(self.cpu, [toks])[0], None
        except Exception as e:  # noqa: BLE001
            return None, f"{err + '; ' if err else ''}cpu: {type(e).__name__}: {e}"

    def parse(self, token_lists: Sequence[Sequence[str]]):
        """Parse sentences. Returns (results, errors), both indexed like the
        input: results[i] = (tree, xpos) or None; errors[i] = message or None."""
        n = len(token_lists)
        results: list = [None] * n
        errors: list = [None] * n
        order = sorted(range(n), key=lambda i: len(token_lists[i]))
        batch: list[int] = []
        size = 0

        def flush(idx: list[int]) -> None:
            if not idx:
                return
            try:
                for i, r in zip(idx, self._run(self.nlp, [token_lists[i] for i in idx])):
                    results[i] = r
                return
            except Exception as e:  # noqa: BLE001
                if _is_oom(e):
                    self.events["oom_batches"] += 1
                    _free_gpu()
            for i in idx:
                results[i], errors[i] = self._one(token_lists[i])

        for i in order:
            L = len(token_lists[i])
            if batch and size + L > self.sub_batch_tokens:
                flush(batch)
                batch, size = [], 0
            batch.append(i)
            size += L
        flush(batch)
        return results, errors


def gpu_available() -> bool:
    if os.environ.get("NEGCONJ_FORCE_CPU"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
