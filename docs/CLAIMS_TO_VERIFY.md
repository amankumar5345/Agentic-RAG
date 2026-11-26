# Claims to verify

Moved out of `README.md` on 2026-11-26. Nothing in this repo produces a stored
result file for these claims: there is no `results/` directory, no benchmark
script, and no committed evaluation output. `evaluate_node` computes
per-response scores at runtime, but those numbers are not a benchmark and are
not published here.

## 1. DeepEval quality metrics table

The README previously presented this table as if the values were measured:

| Metric | Threshold | Purpose |
| :--- | :---: | :--- |
| **Faithfulness** | $\ge 0.50$ | Verifies output is strictly grounded in retrieved documentation |
| **Answer Relevancy** | $\ge 0.50$ | Evaluates answer completeness relative to user intent |
| **Contextual Relevancy** | $\ge 0.50$ | Assesses precision and relevance of retrieved text chunks |

The `0.5` values are configured cut-offs in `src/core/graph.py`
(`FaithfulnessMetric(threshold=0.5, ...)` and the two other metrics), not
measured scores. The "Purpose" column is a description of what DeepEval's
metrics do.

**To verify:** run a seeded benchmark that writes per-example scores to
`results/`, then quote only that file.

## 2. "Automatically evaluates every response"

The README said `evaluate_node` "Automatically evaluates every response via
DeepEval". In code, `evaluate_node` runs after the orchestrator and calls the
three metrics with the DeepSeek judge. It is unconditional, but without
`DEEPSEEK_API_KEY` (or when the API call fails) each metric stores
`{"score": 0.0, "passed": False, "error": ...}` instead of a real score. So the
claim holds only for runs that have a working DeepSeek key and network access.

**To verify:** add a test that mocks the judge, asserts `eval_results` is
populated, and state the condition in the README.

## 3. Pipeline diagram scores

The mermaid diagram labels the evaluation node "Scores: Faithfulness, Answer
Relevancy, Contextual Relevancy". That names the metrics, not results; it is
covered by item 1.

## What would let us put numbers back in the README

1. A script that runs a fixed question set through the graph with a fixed seed
   and temperature 0.
2. Mocked or cached judge calls so reruns are free.
3. Output written to `results/<run>.json` (or `.csv`).
4. A README sentence that cites that file, and updates it only from the file.
