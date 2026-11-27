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

RESULTS_DIR="${LITHO_RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR/dev"

for arm in A0 A1 A2 A3 A4 A5 A6 A7 A8; do
  python -m litho_eval.run \
    --arm "$arm" --split dev \
    --out "$RESULTS_DIR/dev/${arm}.json" \
    $MOCK "${LIMIT[@]}"
  python -m litho_eval.run \
    --arm "$arm" --split test \
    --out "$RESULTS_DIR/${arm}.json" \
    $MOCK "${LIMIT[@]}"
done

python -m litho_eval.compare \
  --baseline A1 \
  --results-dir "$RESULTS_DIR" \
  --out-json "$RESULTS_DIR/comparison.json" \
  --out-md "$RESULTS_DIR/comparison.md"
