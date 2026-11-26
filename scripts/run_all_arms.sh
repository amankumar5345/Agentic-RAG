#!/usr/bin/env bash
# Run every arm on dev, then test, then compare against A1.
set -euo pipefail

MOCK=""
LIMIT=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mock)
      MOCK="--mock"
      shift
      ;;
    --limit)
      LIMIT=(--limit "$2")
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p results/dev

for arm in A0 A1 A2 A3 A4 A5 A6 A7 A8; do
  python -m litho_eval.run \
    --arm "$arm" --split dev \
    --out "results/dev/${arm}.json" \
    $MOCK "${LIMIT[@]}"
  python -m litho_eval.run \
    --arm "$arm" --split test \
    --out "results/${arm}.json" \
    $MOCK "${LIMIT[@]}"
done

python -m litho_eval.compare --baseline A1 --results-dir results
