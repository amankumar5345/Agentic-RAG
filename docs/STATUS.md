# Project status and open blockers

This is the single place to check before quoting any number. Do not put
counts or scores here; read them from `results/` or from the validation
commands below.

## Dataset (main blocker)

- `bench/data/*.jsonl` contains **drafts**, not verified cases.
- Drafting is automated; **verification is human**. There is no way to run the
  evaluation harness on unverified cases without breaking the dataset rule.
- Check the current state:
  ```bash
  python -m bench.validate --include-draft
  ```
- To unblock every evaluation:
  ```bash
  python -m bench.review --reviewer "Your Name"
  python -m bench.validate
  ```
- Known drafting weakness: a small local/cheap model repeats itself at
  temperature 0. Prompt topics/plans now force variation and duplicate IDs are
  rejected, but the final counts must be checked with `bench.validate`.

## RAG arms (A0-A7)

- Implemented in `litho_eval/arms.py`, configured in `configs/arms.yaml`.
- Tests cover every arm offline; no real arm has been run because there are
  zero verified cases.
- A1/A3/A4 real runs need a running Ollama server with `bge-m3:latest` (and A4
  also downloads the FlashRank model). A0 and A2 only need verified cases.
- A5-A7 use the existing agent graph, not `LLMClient`, so their generator cost
  and token counts are recorded as **untracked** (`cost_usd: null`).
- Retrieval metrics are reported only for A1-A4.
- Exact run:
  ```bash
  bash scripts/run_all_arms.sh            # add --mock --limit 3 for smoke
  ```

## Guardrail evaluation (G0-G3 + A5)

- Implemented in `litho_eval/guardrails_eval.py`.
- Blocked by the same zero verified adversarial cases.
- G2 (logistic regression over bge-m3) also needs a running Ollama server.
- G1 and A5 need `GROQ_API_KEY`.
- Cost per query is `null` for G1/A5 because those paths do not go through
  `LLMClient`; G0/G2/G3 report 0.
- Exact run:
  ```bash
  python -m litho_eval.guardrails_eval --split test
  ```

## Prices and models

- `configs/prices.yaml` has Groq list prices for `gpt-oss-120b`,
  `gpt-oss-20b` and `qwen3.8-27b`.
- DeepSeek prices are intentionally blank (peak/off-peak and the account has
  no balance); the harness judge defaults to `groq/qwen/qwen3.8-27b`.
- Real runs fail loudly with `MissingPriceError` if a selected model has no
  price.

## Geometry bridge and end-to-end latency

- Implemented in `litho_eval/geometry_eval.py`; `A5` runs with
  `spawn_geometry` replaced by a capturing stub.
- `scripts/measure_e2e.py --real-unreal` measures 50 text-to-ack runs against
  Unreal Engine 5; fake mode uses `tools/fake_unreal.py`.
- Both are blocked by zero verified geometry cases. A1/A3/A4-style retrieval is
  not needed for geometry, but the agent itself still needs Groq.
- Commands:
  ```bash
  python -m litho_eval.geometry_eval
  python scripts/measure_e2e.py --real-unreal
  ```
- `results/geometry.json` and `results/geometry_failures.md` currently come
  from a **mock** run (`"mock": true`); the failure list is mock data.
- `results/e2e_latency.json` is also a **mock** run (`"mock": true`,
  `ue_mode: "fake"`, 50 synthetic runs). No real timings exist.
- Commands:
  ```bash
  python -m litho_eval.geometry_eval --mock
  python scripts/measure_e2e.py --mock
  python -m litho_eval.geometry_eval          # real, needs verified cases
  python scripts/measure_e2e.py --real-unreal # 50 runs against UE5
  ```
- Every latency result carries `ue_mode`; fake numbers are never labeled
  real.

## Indirect prompt injection

- 40 draft injection cases live in `bench/injection/cases.jsonl` (10 per
  attack type), built from real corpus passages.
- Review them with the same tool:
  ```bash
  python -m bench.review --injection --reviewer "Your Name"
  ```
- `bench/injection/build_collection.py` creates the separate `litho_injected`
  Chroma collection; it never deletes or writes the real collection. Real
  collection building needs Ollama `bge-m3`.
- D0-D3 helpers (wrapper, sanitiser, tool provenance gate) and rule-based
  attack-success detection are implemented and tested.
- Real D0-D3 execution is implemented in `litho_eval.injection`; it needs
  verified injection cases and Groq.
- `results/injection.json` is a **mock** run (`"mock": true`); utility cost is
  still `null` until a real clean-QA utility run exists.
- Commands:
  ```bash
  python -m bench.injection.cases
  python -m bench.injection.build_collection
  python -m litho_eval.injection --mock
  ```

## Citations and A8

- `load_prompt` now returns chunks labelled `[C1]..[Cn]` with document and
  section metadata, and instructs the model to cite every factual/procedural
  claim.
- `src/core/citations.py` validates citation ids, claim coverage and
  per-sentence support, repairs once, then abstains with the closest sources.
  `CitationAnswer.to_dict()` is the `{text, citations}` shape for the UI.
- `configs/arms.yaml` defines A8 (`agent_citations`). A8 is wired into
  `litho_eval.runner` with a capturing retriever, citation generator and judge;
  the Assistant live path uses A8 and shows structured chunks/citations.
- `results/citations.json` is a blocked/mock file until real A5/A8 results
  exist; `--mock` writes a clearly labelled mock table.
- Tests cover missing citation, bogus id, unsupported claim, correct answer,
  repair, abstention, stable `[Cn]` ids and the A8 wrapper.
- Commands:
  ```bash
  python -m litho_eval.citations_eval --mock
  python -m litho_eval.run --arm A8 --split test --out results/A8.json
  python -m litho_eval.citations_eval
  ```

## Reporting, judge validation and CI

- `scripts/export_judge_sample.py` exports 60 stratified answerable cases to
  `bench/judge_sample.csv` with `human_label` blank. It refuses mock results.
- `scripts/judge_agreement.py` joins the filled CSV to the judge's labels and
  writes `results/judge_validation.json`; it fails below 50 labels.
- `scripts/build_readme.py` fills `<!-- RESULTS:name -->` blocks; missing, mock
  or blocked files render `NOT YET RUN`. `--check` fails on stale blocks.
- `scripts/regression_gate.py` compares a fresh cached-results file with
  `results/baseline.json` using `configs/gate.yaml`; baseline is not committed
  until a real run exists.
- CI runs ruff, pytest, mock arms, README check and claim check.
- Commands:
  ```bash
  python scripts/export_judge_sample.py
  python scripts/judge_agreement.py
  python scripts/build_readme.py --check
  python scripts/regression_gate.py
  ```

## Streamlit UI

- Multipage app under `src/ui/`, default **demo** mode; live mode needs Groq,
  DeepSeek and Ollama.
- Demo pages replay `results/*.json`; mock or blocked files are labelled
  MOCK / NOT YET RUN.
- Assistant live trace currently captures the answer, role, guardrail flag,
  retrieval-call count and total latency; structured chunks and citation
  validator state are only available in recorded/demo traces.
- Live wiring is complete for the Assistant and Arm Playground (A8 trace and
  real arm runs). Safety Lab and Geometry Lab text-to-commands still use
  recorded/rule-based demo logic.
- `AppTest` covers every page in demo mode; unit tests cover loaders and chart
  builders.
- Docker and Community Cloud setup are in `Dockerfile`, `.streamlit/` and
  `docs/DEMO.md`.
- Commands:
  ```bash
  streamlit run src/ui/app.py
  LITHO_MODE=demo streamlit run src/ui/app.py
  python -m pytest tests/test_ui_pages.py
  ```

## Deploy readiness

- Streamlit server smoke: `/_stcore/health` returned 200 on a fresh port.
- Demo mode needs no keys, Ollama or Unreal.
- Dockerfile and `.streamlit` config are present; the local Docker daemon was
  not running, so the image was not built here.
- `requirements.txt` pins were read under Python 3.11.4 while the project and
  CI target 3.12; re-pin from 3.12 before release.
- Safety Lab and Geometry Lab text-to-commands still use demo/rule-based paths;
  Assistant and Arm Playground have live A8/A0-A8 paths.

## Claim checking

- `python scripts/claim_check.py` passes: no quantified claim outside
  `docs/CLAIMS_TO_VERIFY.md` is currently unbacked.
- `docs/CLAIMS_TO_VERIFY.md` is exempt as a record of old claims; if those
  numbers are copied back into README, the checker will flag them.
- `results/` contains mock guardrail, geometry and e2e output. `claim_check`
  ignores mock results, so they cannot back a README claim. There are no real
  evaluation numbers yet.

## Other known limitations

- The unanswerable check is a keyword-window heuristic; a reviewer must
  confirm.
- Geometry `replay()` treats `polygon` as one edge because the command schema
  has no vertex list.
- Drafting at temperature 0 with a small model can still duplicate text even
  with distinct prompts; the dedupe layer prevents duplicate IDs, not
  duplicate wording.
