#!/usr/bin/env bash
# Pull results from the Modal Volume and verify every checksum locally.
#   scripts/pull_results.sh            # full run: /data/results -> ./results
#   scripts/pull_results.sh pilot      # /data/pilot/results -> ./results_pilot
set -euo pipefail
cd "$(dirname "$0")/.."
MODAL="${MODAL:-.venv/bin/modal}"
run="${1:-full}"
if [[ "$run" == "full" ]]; then src="results"; dst="./results"; else src="$run/results"; dst="./results_$run"; fi

mkdir -p "$dst"
"$MODAL" volume get --force negconj-data "$src" "$dst.tmp"
# `volume get` of a directory nests it under the destination; flatten.
if [[ -d "$dst.tmp/results" ]]; then rsync -a "$dst.tmp/results/" "$dst/"; else rsync -a "$dst.tmp/" "$dst/"; fi
rm -rf "$dst.tmp"

cd "$dst"
shasum -a 256 -c CHECKSUMS.sha256 --quiet && echo "all $(wc -l < CHECKSUMS.sha256 | tr -d ' ') checksums OK in $dst"
