# Reproduce

## Hardware used for local development

- Windows host, NVIDIA GTX GPU with 4 GB VRAM, 16 GB class RAM.
- About 84 GB free disk at the time of the last local run.
- Local interpreter: Python 3.11.4 (the project requires the version declared in
  `pyproject.toml`; CI uses 3.12).
- Ollama must be running for embedding and retrieval:
  ```bash
  ollama pull bge-m3:latest
  ```

## Commands

```bash
# install
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# tests and lint
make test
make lint
python scripts/claim_check.py

# dataset
python -m bench.validate --include-draft
python -m bench.review --reviewer "Your Name"
python -m litho_eval.guardrails_eval
python -m litho_eval.geometry_eval
python scripts/measure_e2e.py              # fake ack
python scripts/measure_e2e.py --real-unreal
python -m litho_eval.injection --mock
python -m litho_eval.citations_eval --mock

# arms
bash scripts/run_all_arms.sh               # add --mock --limit 5 for CI

# judge validation
python scripts/export_judge_sample.py
# fill human_label in bench/judge_sample.csv by hand
python scripts/judge_agreement.py

# README blocks
python scripts/build_readme.py
python scripts/build_readme.py --check
```

## Runtime

- Mock/CI path (lint, tests, mock arms, README check, claim check) is on the
  order of tens of seconds to a few minutes on the hardware above.
- A real arm run is dominated by retrieval and generation calls. No real
  end-to-end runtime has been measured in this checkout; `results/e2e_latency.json`
  is a mock artifact until `--real-unreal` is run.
- Drafting was rate-limited by the provider quota; it is not a compute-bound
  benchmark.

## API cost estimate

Prices are read from `configs/prices.yaml`. The formula is:

```text
cost = prompt_tokens / 1_000_000 * input_per_million
     + completion_tokens / 1_000_000 * output_per_million
```

Illustrative calculation using the checked-in Groq list prices and one
million prompt tokens plus a quarter million completion tokens:

```text
qwen3.8-27b:
  1.00 * 0.80 + 0.25 * 4.00 = 1.80 USD

gpt-oss-120b:
  1.00 * 0.15 + 0.25 * 0.60 = 0.30 USD
```

Real totals require the per-run token counts written by `litho_eval.client`
into its cache. Do not quote an estimate without those counts.

## Notes

- Real Unreal latency is a separate mode (`--real-unreal`) and needs a running
  UE project. Fake ack results are never presented as real.
- Judge agreement needs at least 50 hand labels; the script fails otherwise.
