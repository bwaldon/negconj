"""Modal app: image, Volume, stage functions and orchestrator (work plan §4–§5).

    modal deploy negconj/modal_app.py      # then scripts/launch.py spawns `orchestrate`

Runs live under a root on the Volume: ``/data`` for the full run, ``/data/{run}``
for others (e.g. ``/data/pilot``). Raw XML is shared under ``/data/raw``.
Every stage is skipped when its output exists, so relaunching resumes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "negconj"
VOLUME_NAME = "negconj-data"
DATA = Path("/data")
STANZA_DIR = "/opt/stanza_resources"
MAX_GPU_CONTAINERS = 10  # Starter plan GPU concurrency; v2 Volume

PINNED = [
    "beautifulsoup4==4.15.0", "lxml==6.1.3", "pandas==3.0.6", "pyarrow==26.0.0",
    "stanza==1.15.0", "transformers==5.20.2", "nltk==3.10.3", "tqdm==4.70.1",
]
HF_HOME = "/opt/hf"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*PINNED)
    .env({"STANZA_RESOURCES_DIR": STANZA_DIR, "HF_HOME": HF_HOME})
    # Every model in parse.PARSER_CONFIGS, plus ELECTRA's weights, baked into the image.
    .run_commands(
        "python -c \"import stanza; "
        "stanza.download('en', processors='tokenize,mwt,pos,constituency', model_dir='" + STANZA_DIR + "'); "
        "stanza.download('en', processors={'pos': 'combined_electra-large', "
        "'constituency': 'ptb3-revised_electra-large'}, model_dir='" + STANZA_DIR + "'); "
        "from transformers import AutoModel, AutoTokenizer; "
        "AutoTokenizer.from_pretrained('google/electra-large-discriminator'); "
        "AutoModel.from_pretrained('google/electra-large-discriminator')\"")
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_python_source("negconj")
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
VOLS = {str(DATA): vol}


def run_root(run: str) -> Path:
    return DATA if run == "full" else DATA / run


def _state_path(root: Path) -> Path:
    return root / "orchestrator.json"


def _load_state(root: Path) -> dict:
    p = _state_path(root)
    return json.loads(p.read_text()) if p.exists() else {"stages": {}, "shards": {}}


def _save_state(root: Path, state: dict) -> None:
    from negconj.shards import atomic_write_bytes
    atomic_write_bytes(_state_path(root), json.dumps(state, indent=1, default=str).encode())
    vol.commit()


# --------------------------------------------------------------------------
# Stage functions
# --------------------------------------------------------------------------

@app.function(volumes=VOLS, cpu=2, memory=4096, timeout=3600, nonpreemptible=True)
def download(release: str | None = None, titles: list[str] | None = None) -> dict:
    """Fetch the release point (all titles, or the named titles) into /data/raw."""
    from negconj import download as dl
    vol.reload()
    current = dl.current_release_point()
    release = release or current
    if release is None:
        raise RuntimeError("could not determine the current release point from download.html")
    raw = DATA / "raw"
    tag = "All" if not titles else "+".join(titles)
    rec_path = raw / f"release_{release}_{tag}.json"
    if rec_path.exists():
        return json.loads(rec_path.read_text())
    rec = {"release": release, "current_on_site": current, "zips": {}, "xml": []}
    for t in (titles or ["All"]):
        url = dl.zip_url(release, t)
        z = dl.fetch(url, raw / "zips" / Path(url).name)
        rec["zips"][Path(url).name] = {"url": url, "sha256": dl.sha256(z)}
        rec["xml"] += [p.name for p in dl.unzip_xml(z, raw / release)]
    rec["xml"] = sorted(set(rec["xml"]))
    rec_path.write_text(json.dumps(rec, indent=1))
    vol.commit()
    return rec


# Pilot: peak RSS 2.4 GiB for Title 42 (the largest), 1.3 GiB for Title 26.
@app.function(volumes=VOLS, cpu=2, memory=8192, timeout=3600, nonpreemptible=True)
def extract_title(release: str, xml_name: str, run: str) -> dict:
    import resource
    from negconj.stages import extract_stage
    vol.reload()
    t0 = time.monotonic()
    out = extract_stage(DATA / "raw" / release / xml_name, run_root(run))
    vol.commit()
    out["seconds"] = time.monotonic() - t0
    out["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return out


@app.function(volumes=VOLS, cpu=2, memory=16384, timeout=3600, nonpreemptible=True)
def plan_shards(run: str, target_chars: int, titles: list[str] | None = None) -> dict:
    from negconj.stages import plan_stage
    vol.reload()
    plan = plan_stage(run_root(run), target_chars, titles)
    vol.commit()
    return plan


_MODELS: dict = {}


@app.function(volumes=VOLS, gpu="L4", timeout=2 * 3600, retries=3,
              max_containers=MAX_GPU_CONTAINERS)
def process_shard(run: str, shard_id: str, min_matches: int = 32) -> dict:
    """Preemptible GPU worker; resumes from /progress on restart."""
    import pandas as pd
    import torch
    from negconj.parse import Parser
    from negconj.segment import build_tokenizer, stanza_tokenizer
    from negconj.shards import prefilter, run_shard, shard_title
    t0 = time.monotonic()
    vol.reload()
    root = run_root(run)
    if not _MODELS:  # once per container
        _MODELS["tok"] = stanza_tokenizer(build_tokenizer(use_gpu=True))
        _MODELS["parser"] = Parser(use_gpu=True)
    plan = json.loads((root / "shards.json").read_text())
    units = prefilter(pd.read_parquet(root / "units" / f"{shard_title(shard_id)}.parquet"))
    counts = run_shard(shard_id, plan[shard_id], units, root, _MODELS["tok"], _MODELS["parser"],
                       min_matches=min_matches, commit=vol.commit)
    counts["gpu_seconds"] = time.monotonic() - t0  # GPU is held for the whole call
    counts["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return counts


@app.function(volumes=VOLS, cpu=4, memory=32768, timeout=3 * 3600, nonpreemptible=True)
def merge(run: str) -> dict:
    from negconj.stages import merge_stage
    vol.reload()
    root = run_root(run)
    out = merge_stage(root, json.loads((root / "shards.json").read_text()))
    vol.commit()
    return out


@app.function(volumes=VOLS, cpu=4, memory=32768, timeout=3 * 3600, nonpreemptible=True)
def classify(run: str) -> dict:
    from negconj.stages import classify_stage
    vol.reload()
    out = classify_stage(run_root(run))
    vol.commit()
    return out


@app.function(volumes=VOLS, cpu=4, memory=32768, timeout=3 * 3600, nonpreemptible=True)
def report(run: str, manifest: dict, stage_counts: dict) -> dict:
    from negconj.report import report_stage
    vol.reload()
    root = run_root(run)
    out = report_stage(root, json.loads((root / "shards.json").read_text()), manifest, stage_counts)
    vol.commit()
    return out


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def _shard_complete(root: Path, shard_id: str, sections: list[str]) -> bool:
    from negconj.shards import read_done
    return set(sections) <= set(read_done(root, shard_id))


@app.function(volumes=VOLS, cpu=1, memory=4096, timeout=24 * 3600, nonpreemptible=True)
def orchestrate(run: str = "full", release: str | None = None, titles: list[str] | None = None,
                target_chars: int = 400_000, gpu_budget_seconds: float = 36_000,
                git_commit: str = "unknown", min_matches: int = 32,
                stop_after: str | None = None) -> dict:
    """Run every stage in order; skip what is done; stop launching shards
    once reported GPU seconds pass ``gpu_budget_seconds``. ``stop_after="extract"``
    returns after extraction (per-title prefilter sizes, for cost projection)."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    from negconj.shards import title_sort_key

    t_start = time.monotonic()
    vol.reload()
    root = run_root(run)
    root.mkdir(parents=True, exist_ok=True)
    state = _load_state(root)
    state.update({"run": run, "git_commit": git_commit, "target_chars": target_chars,
                  "gpu_budget_seconds": gpu_budget_seconds})

    def log(msg: str) -> None:
        print(f"[orchestrate {run}] {msg}", flush=True)
        state.setdefault("log", []).append(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}")
        _save_state(root, state)

    # 1. download
    rec = download.remote(release, titles)
    state["stages"]["download"] = rec
    release = rec["release"]
    log(f"release {release} ({len(rec['xml'])} XML files); site current = {rec['current_on_site']}")
    xmls = [x for x in rec["xml"]
            if not titles or x.removeprefix("usc").removesuffix(".xml") in titles]

    # 2. extract (parallel, one call per title)
    ex = list(extract_title.map([release] * len(xmls), xmls, [run] * len(xmls)))
    state["stages"]["extract"] = {e["title"]: e for e in ex}
    log(f"extracted {len(ex)} titles, {sum(e['units'] for e in ex)} units, "
        f"{sum(e['prefilter_chars'] for e in ex)} prefilter chars")
    all_titles = sorted((e["title"] for e in ex), key=title_sort_key)
    if stop_after == "extract":
        return {"status": "stopped_after_extract", "release": release, "zips": rec["zips"],
                "extract": state["stages"]["extract"]}

    # 3. plan
    plan = plan_shards.remote(run, target_chars, all_titles)
    log(f"plan: {len(plan)} shards, {sum(len(v) for v in plan.values())} sections")

    # 4. process shards, at most MAX_GPU_CONTAINERS in flight, within budget
    vol.reload()
    shard_stats = state.setdefault("shards", {})
    gpu_total = sum(s.get("gpu_seconds", 0) for s in shard_stats.values())
    todo = [s for s, secs in plan.items() if not _shard_complete(root, s, secs)]
    log(f"{len(todo)} shards to process; GPU seconds so far {gpu_total:.0f}")
    stopped = False
    with ThreadPoolExecutor(MAX_GPU_CONTAINERS) as pool:
        inflight = {}
        queue = list(todo)
        while queue or inflight:
            while queue and len(inflight) < MAX_GPU_CONTAINERS and not stopped:
                if gpu_total >= gpu_budget_seconds:
                    stopped = True
                    log(f"GPU budget reached ({gpu_total:.0f}s); not launching "
                        f"{len(queue)} remaining shards, first {queue[0]}")
                    break
                sid = queue.pop(0)
                inflight[pool.submit(process_shard.remote, run, sid, min_matches)] = sid
            if stopped:
                queue = []
            if not inflight:
                break
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for f in finished:
                sid = inflight.pop(f)
                try:
                    c = f.result()
                except Exception as e:  # noqa: BLE001 — recorded; relaunch resumes it
                    shard_stats[sid] = {**shard_stats.get(sid, {}), "error": repr(e)}
                    log(f"shard {sid} failed: {e!r}")
                    continue
                prev = shard_stats.get(sid, {}).get("gpu_seconds", 0)
                c["gpu_seconds"] += prev
                shard_stats[sid] = c
                gpu_total += c["gpu_seconds"] - prev
                log(f"shard {sid}: {c['matches']} matches, {c['failed']} failed, "
                    f"{c['gpu_seconds']:.0f}s; total GPU {gpu_total:.0f}s")
    vol.reload()
    incomplete = [s for s, secs in plan.items() if not _shard_complete(root, s, secs)]
    if incomplete:
        log(f"{len(incomplete)} shards incomplete (e.g. {incomplete[0]}); stopping before merge. "
            "Relaunch to resume.")
        return {"status": "incomplete", "incomplete": incomplete, "gpu_seconds": gpu_total}

    # 5. merge, 6. classify, 7. report
    state["stages"]["merge"] = merge.remote(run)
    log(f"merge: {state['stages']['merge']}")
    state["stages"]["classify"] = classify.remote(run)
    log(f"classify: {state['stages']['classify']}")

    m, c = state["stages"]["merge"], state["stages"]["classify"]
    stage_counts = {
        "titles": len(all_titles),
        "units": sum(e["units"] for e in ex),
        "units_no_section": sum(e["no_section"] for e in ex),
        "prefilter_sections": sum(len(v) for v in plan.values()),
        "shards": len(plan),
        "raw_sentence_splits": m["raw_sentence_splits"],
        "re_merged_splits": m["re_merged_splits"],
        "segmented_sentences": m["segmented_sentences"],
        "regex_matches": m["matches"],
        "parsed": m["parses"],
        "parse_failures": m["failures"],
        "duplicate_rows_dropped_in_merge": m["duplicate_parse_rows_dropped"],
        "not_and_pairs": c["pairs"],
        "flagged": c["flagged"],
        "flagged_relaxed": c["flagged_relaxed"],
    }
    gpus = sorted({s.get("gpu") for s in shard_stats.values() if s.get("gpu")})
    import pandas as pd
    vol.reload()
    p = pd.read_parquet(root / "results" / "parses.parquet", columns=["stanza_version", "model_package"])
    manifest = {
        "release_point": release,
        "zips": rec["zips"],
        "git_commit": git_commit,
        "stanza_version": sorted(p["stanza_version"].unique().tolist()),
        "model_package": sorted(p["model_package"].unique().tolist()),
        "gpu_type": gpus,
        "gpu_seconds": gpu_total,
        "wall_seconds_this_launch": time.monotonic() - t_start,
        "target_chars": target_chars,
        "min_matches": min_matches,
        "extract": state["stages"]["extract"],
        "shards": shard_stats,
    }
    checks = report.remote(run, manifest, stage_counts)
    state["stages"]["report"] = checks
    log(f"report: checks ok = {checks.get('ok')}")
    return {"status": "done", "counts": stage_counts, "checks": checks, "gpu_seconds": gpu_total}


@app.function(volumes=VOLS, cpu=1, memory=2048, timeout=6 * 3600, nonpreemptible=True)
def measure_extract(release: str, titles: list[str]) -> list[dict]:
    """Pilot step 2: extract large titles alone to size extract_title's memory."""
    download.remote(release, titles)
    xmls = [f"usc{t}.xml" for t in titles]
    return list(extract_title.map([release] * len(xmls), xmls, ["measure"] * len(xmls)))
