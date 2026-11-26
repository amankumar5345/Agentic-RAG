# Limitations

- LLM-drafted questions share vocabulary with their source passages, which
  flatters lexical and dense retrieval.
- The corpus is one collection of documents, dominated by laser-safety
  material and vendor manuals; results do not generalise to other cleanrooms.
- One generator model family is evaluated. Other models may follow injected
  instructions differently.
- Sample sizes are small; confidence intervals will be wide.
- The judge is itself a language model and can be biased toward its own
  outputs or toward fluent answers.
- End-to-end latency has two modes: fake Unreal acknowledgements and real
  Unreal Engine runs. Fake numbers are only for CI and are never real
  measurements.
- There is no user study; usability and trust are not measured.
