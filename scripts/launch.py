"""Deploy the Modal app and spawn the orchestrator (runs detached from the laptop).

    python scripts/launch.py --run pilot --titles 29 18 --gpu-budget-seconds 3600
    python scripts/launch.py --run full --gpu-budget-seconds <approved budget>
    python scripts/launch.py --measure-extract 26 42

Relaunching with the same arguments resumes: every stage skips finished work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "negconj"], cwd=ROOT).returncode
    if dirty:
        sys.exit("negconj/ has uncommitted changes; commit first so the run is reproducible.")
    return sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot", help="'full' writes to /data, else /data/<run>")
    ap.add_argument("--titles", nargs="*", default=None, help="restrict to these titles")
    ap.add_argument("--release", default=None, help="release point, e.g. 119-121 (default: current)")
    ap.add_argument("--target-chars", type=int, default=400_000)
    ap.add_argument("--min-matches", type=int, default=256)
    ap.add_argument("--gpu-budget-seconds", type=float, required=False, default=3600)
    ap.add_argument("--measure-extract", nargs="*", default=None,
                    help="only extract these titles to measure memory/runtime")
    ap.add_argument("--no-deploy", action="store_true")
    a = ap.parse_args()

    commit = git_commit()
    if not a.no_deploy:
        subprocess.run([sys.executable, "-m", "modal", "deploy", "negconj/modal_app.py"],
                       cwd=ROOT, check=True)
    if a.measure_extract:
        fn = modal.Function.from_name("negconj", "measure_extract")
        call = fn.spawn(a.release or "119-121", a.measure_extract)
    else:
        fn = modal.Function.from_name("negconj", "orchestrate")
        call = fn.spawn(run=a.run, release=a.release, titles=a.titles,
                        target_chars=a.target_chars, gpu_budget_seconds=a.gpu_budget_seconds,
                        git_commit=commit, min_matches=a.min_matches)
    print(f"spawned {fn.object_id if hasattr(fn, 'object_id') else ''} call {call.object_id} "
          f"(commit {commit[:10]}). Follow with: modal app logs negconj")
    (ROOT / "data").mkdir(exist_ok=True)
    with open(ROOT / "data" / "launches.log", "a") as fh:
        fh.write(f"{call.object_id} {commit} {vars(a)}\n")


if __name__ == "__main__":
    main()
