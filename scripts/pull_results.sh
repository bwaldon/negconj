#!/usr/bin/env bash
# Pull results from the Modal Volume and verify every checksum locally.
#   scripts/pull_results.sh            # full run: /data/results -> ./results
#   scripts/pull_results.sh pilot      # /data/pilot/results -> ./results_pilot
# Needs Modal credentials in the environment (e.g. `source scripts/modal_env.sh`).
set -euo pipefail
cd "$(dirname "$0")/.."
repo="$PWD"
MODAL="${MODAL:-$repo/.venv/bin/modal}"
run="${1:-full}"
if [[ "$run" == "full" ]]; then src="results"; dst="$repo/results"; else src="$run/results"; dst="$repo/results_$run"; fi

# `modal volume get VOL path/to/results` (no destination) writes ./results in the cwd.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
(cd "$tmp" && "$MODAL" volume get negconj-data "$src" >/dev/null)
[[ -f "$tmp/results/CHECKSUMS.sha256" ]] || { echo "no CHECKSUMS.sha256 in $src" >&2; exit 1; }
rm -rf "$dst"
mv "$tmp/results" "$dst"

cd "$dst"
shasum -a 256 -c CHECKSUMS.sha256 --quiet && echo "all $(wc -l < CHECKSUMS.sha256 | tr -d ' ') checksums OK in $dst"
