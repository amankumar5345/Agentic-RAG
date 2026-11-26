#!/usr/bin/env bash
# Optional index sweep. Disabled by default; run this explicitly.
set -euo pipefail

MOCK=""
LIMIT=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock) MOCK="--mock"; shift ;;
    --limit) LIMIT=(--limit "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for chunk in 500 1000 1500; do
  overlap=$((chunk * 15 / 100))
  for embedding in bge-m3:latest nomic-embed-text:latest; do
    out_dir="results/sweeps/${embedding//[:\\/]/_}/${chunk}"
    mkdir -p "$out_dir"
    for arm in A1 A2 A3 A4; do
      python -m litho_eval.run \
        --arm "$arm" --split test \
        --chunk-size "$chunk" --chunk-overlap "$overlap" \
        --embedding-model "$embedding" \
        --out "$out_dir/${arm}.json" \
        $MOCK "${LIMIT[@]}"
    done
  done
done
