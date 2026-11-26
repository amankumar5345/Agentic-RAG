# LithoBench dataset card

## What this is

LithoBench is the labelled evaluation set for the Litho Agentic RAG project.
It is split by behaviour, not by module:

- **QA** — answerable technical questions with quoted gold answers and
  character-offset sources, plus deliberately unanswerable questions.
- **Adversarial** — prompts that should be blocked, with benign controls that
  should be answered.
- **Geometry** — drawing utterances with gold command lists, including
  utterances that must trigger a size clarification.
- **Router** — queries labelled with the `RoleMaker` route.

It is **not** a benchmark result. Only `bench/validate.py` and future
evaluation scripts may turn it into numbers, and only from `verified` cases.

## Composition (targets)

| Type | Target | Breakdown |
| :--- | ---: | :--- |
| QA | 180 | 60 single_hop, 40 multi_hop, 25 numeric, 25 hazard, 30 unanswerable |
| Adversarial | 80 | 15 jailbreak, 10 off_topic, 10 harmful_procedure, 10 injection_direct, 5 gibberish, 30 benign_control |
| Geometry | 50 | single, multi_step, loop, synonym, angle_words |
| Router | 60 | 15 per role, including the code's exact `Maintainance Agent` spelling |

"QA 150 plus 30 unanswerable" is read here as **180 QA cases total**, with the
30 unanswerable cases in their own subtype. `bench/schema.py` is the source of
truth for these targets.

Counts for the data actually checked in must be read from:

```bash
python -m bench.validate
```

The checked-in `bench/data/*.jsonl` files contain drafts; run
`python -m bench.validate --include-draft` for the current counts. Nothing is
verified until `bench.review` has been run.

## Origins and the human rule

Every case records `origin`:

| Origin | Meaning |
| :--- | :--- |
| `human` | Written by a person, no model draft |
| `llm_draft` | Model-drafted and accepted unchanged |
| `llm_edited` | Model-drafted and substantially changed during review |

**Rule: at least 50% of QA cases must be human-written or substantially
human-edited** (`origin` in `human` or `llm_edited`). `bench/validate.py`
fails the dataset when this is not met.

## How it was built

1. `bench/draft_cases.py` reads `our_database/all_contents.txt` and splits it
   with the same `RecursiveCharacterTextSplitter` settings as
   `our_database/load_txt.py` (1000 characters, 150 overlap).
2. Passages are sampled with a fixed seed. The drafting model is configurable
   and is checked against the generator under test (`openai/gpt-oss-120b`);
   the default drafter is `deepseek/deepseek-chat`, a different family.
3. Every model call goes through `litho_eval/client.py`, which caches on
   provider + model + temperature + seed + max tokens + messages hash. Calls
   use temperature 0 and a fixed seed, so reruns are free and deterministic.
   Real calls need prices in `configs/prices.yaml`; a missing price fails loudly.
4. Drafts are written as `status='draft'`, `origin='llm_draft'` with no
   reviewer. Unanswerable drafts pass a keyword co-occurrence check against
   the corpus before they are kept.
5. A human reviews every draft in `bench/review.py`: accept, edit or reject.
   Accept/edit set `status='verified'`, `reviewer` and `reviewed_at` (UTC);
   edits set `origin='llm_edited'`.
6. Splits are assigned with a fixed seed: 30% `dev`, 70% `test`. QA cases are
   grouped by shared source passage (union-find over all cited passages), so a
   passage is never on both sides. `test` is frozen for final results.

## Review process

- Only `verified` cases are loaded for evaluation by default.
- `--include-draft` exists on `bench/validate.py` for smoke tests only; output
  is labelled `SMOKE`.
- Rejected cases are removed from the dataset and appended to
  `bench/data/rejected.jsonl` for audit.
- A validating source hash (`text_hash`) lets review spot a corpus change.

## Known biases

- **Vocabulary overlap.** LLM-drafted questions reuse the wording of their
  source passage. This flatters lexical and dense retrieval and can overstate
  real-world performance. The human/edited quota is the main mitigation.
- **Corpus imbalance.** The corpus contains many laser-safety SOPs and
  vendor manuals; questions about those topics will dominate unless sampling
  is stratified by document.
- **Duplication.** Several source PDFs are near-duplicates; duplicated
  passages can produce near-duplicate questions. `bench/validate.py` checks
  exact duplicate text only, not paraphrase.
- **Single language.** All cases are English; Hindi translation is untested.
- **Heuristic unanswerable check.** "Absent from corpus" is a keyword-window
  heuristic, not proof; a reviewer must confirm.
- **Geometry labels** describe the current bridge semantics (absolute angles,
  polygon as one edge in `replay`), not a general CAD standard.
- **Router labels** inherit the `Maintainance Agent` spelling and the current
  four-role taxonomy.

## Inter-annotator agreement

_To be filled in after a second person labels 30 items._

- Sample: 30 items (stratified across the four case types), double-labelled.
- Agreement metric: Cohen's kappa (or percentage agreement where the label is
  free text).
- Result: TODO.
- Disagreements resolved by: TODO.

Do not add a number here until the second labelling has happened.

## Reproducing

```bash
# 1. Draft (needs a key for the drafting provider, e.g. DEEPSEEK_API_KEY)
python -m bench.draft_cases --model deepseek/deepseek-chat --seed 13

# 2. Review interactively
python -m bench.review --reviewer "Your Name"

# 3. Validate verified cases (use --include-draft only for smoke tests)
python -m bench.validate

# Tests (offline, mocked LLM)
python -m pytest tests/test_bench_*.py
```
