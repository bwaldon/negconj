"""File-level pipeline stages shared by the local runner and the Modal app.

Every stage reads and writes under one root directory (``/data`` on the
Modal Volume, ``data/`` locally) and is skipped when its output exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import uslm_extract
from .classify import classify_frame
from .shards import (atomic_write_bytes, atomic_write_parquet, merge_shards,
                     plan_shards, prefilter, title_sort_key)


def title_of(xml: Path) -> str:
    return uslm_extract._title_from_path(Path(xml))[0]


def extract_stage(xml: Path, root: Path, conn: str = "and") -> dict:
    root = Path(root)
    title = title_of(xml)
    u_out, e_out = root / "units" / f"{title}.parquet", root / "elements" / f"{title}.parquet"
    if not (u_out.exists() and e_out.exists()):
        elements, units = uslm_extract.extract_title(xml)
        atomic_write_parquet(elements, e_out)
        atomic_write_parquet(units, u_out)  # written last: its presence marks completion
    units = pd.read_parquet(u_out)
    pre = prefilter(units, conn)
    return {"title": title, "units": len(units),
            "units_by_source": units["elem_source"].value_counts().to_dict(),
            "no_section": int(units["section"].isna().sum()),
            "prefilter_units": len(pre),
            "prefilter_sections": int(pre["section_key"].nunique()),
            "prefilter_chars": int(pre["text_parse"].str.len().sum())}


def titles_on_disk(root: Path) -> list[str]:
    return sorted((p.stem for p in (Path(root) / "units").glob("*.parquet")), key=title_sort_key)


def load_units(root: Path, titles: list[str] | None = None) -> pd.DataFrame:
    titles = titles or titles_on_disk(root)
    return pd.concat([pd.read_parquet(Path(root) / "units" / f"{t}.parquet") for t in titles],
                     ignore_index=True)


def load_prefiltered(root: Path, titles: list[str] | None = None,
                     conn: str = "and") -> dict[str, pd.DataFrame]:
    titles = titles or titles_on_disk(root)
    return {t: prefilter(pd.read_parquet(Path(root) / "units" / f"{t}.parquet"), conn) for t in titles}


def plan_stage(root: Path, target_chars: int, titles: list[str] | None = None,
               conn: str = "and") -> dict[str, list[str]]:
    path = Path(root) / "shards.json"
    if path.exists():
        return json.loads(path.read_text())
    plan = plan_shards(load_prefiltered(root, titles, conn), target_chars)
    atomic_write_bytes(path, json.dumps(plan, indent=1).encode())
    return plan


def merge_stage(root: Path, plan: dict[str, list[str]], conn: str = "and") -> dict:
    titles = sorted({sid.rsplit("-", 1)[0] for sid in plan}, key=title_sort_key)
    pre = pd.concat(load_prefiltered(root, titles, conn).values(), ignore_index=True)
    return merge_shards(root, plan, pre)


def classify_stage(root: Path, conn: str = "and") -> dict:
    """pairs, sentences_classified (all matches) and flagged, in results/."""
    root = Path(root)
    res = root / "results"
    parses = pd.read_parquet(res / "parses.parquet")
    pairs, sents = classify_frame(parses, conn)
    titles = sorted(parses["title"].unique(), key=title_sort_key)
    units = load_units(root, titles)[["unit_id", "identifier", "elem_source", "is_appendix",
                                      "text_display"]]
    sents = sents.merge(units, on="unit_id", how="left", validate="many_to_one")
    atomic_write_parquet(pairs, res / "pairs.parquet")
    atomic_write_parquet(sents, res / "sentences_classified.parquet")
    flagged = sents[sents["flagged"]]
    cols = ["sent_id", "title", "section", "identifier", "elem_source", "is_appendix",
            "text", "text_display", "tree", "flagged_relaxed"]
    atomic_write_parquet(flagged[cols], res / "flagged.parquet")
    tmp = res / ".flagged.csv.tmp"
    flagged[cols].to_csv(tmp, index=False)
    tmp.replace(res / "flagged.csv")
    return {"sentences_classified": len(sents), "pairs": len(pairs),
            "flagged": int(sents["flagged"].sum()),
            "flagged_relaxed": int(sents["flagged_relaxed"].sum())}
