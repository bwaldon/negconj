#!/usr/bin/env bash
# Print the orchestrator's new log lines for a run until it finishes or reports a stop.
#   scripts/watch_run.sh [run]      # run: full (default) | disj | pilot
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/modal_env.sh
run="${1:-full}"; path="orchestrator.json"; [[ "$run" != "full" ]] && path="$run/orchestrator.json"
seen=-1
while true; do
  out="$(.venv/bin/modal volume get negconj-data "$path" - 2>/dev/null | .venv/bin/python -c '
import json, sys
s, _ = json.JSONDecoder().raw_decode(sys.stdin.read())
print("\n".join(s.get("log", [])))' 2>/dev/null)"
  n=$(printf "%s\n" "$out" | grep -c .)
  if (( seen < 0 )); then printf "%s\n" "$out" | tail -n 3; seen=$n; continue; fi
  if (( n > seen )); then
    new="$(printf "%s\n" "$out" | tail -n $((n - seen)))"; seen=$n
    printf "%s\n" "$new"
    if printf "%s\n" "$new" | grep -qE "report: checks ok|stopping before merge|GPU budget reached"; then exit 0; fi
  fi
  sleep 60
done
