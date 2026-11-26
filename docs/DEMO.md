# 60-second demo script

1. Start the app:
   ```bash
   LITHO_MODE=demo streamlit run src/ui/app.py
   ```
   The DEMO banner appears; no keys, Ollama or Unreal needed.

2. **Benchmark** — show the arm table with CIs, the correctness/recall bars,
   the cost-vs-p95 scatter, then open one guardrail confusion matrix. Point out
   every block is labelled **MOCK** or **NOT YET RUN** until a real run exists.

3. **Dataset Explorer** — filter to `qa / test / draft`, open one multi-hop
   case and show the gold source passage. Mention the dataset card at the
   bottom.

4. **Geometry Lab** — pick a case, show the JSON, validation result and 2D
   preview against the dashed gold outline. Press **Send to Unreal**; it writes
   the mailbox packet and shows a **SIMULATED** ack.

5. **Safety Lab** — pick a `jailbreak` sample, switch G0 → G3 and D0 → D3,
   and show the rule-based decision trace. Scroll to the poisoned-document
   pair to show the injection text and the sanitised context.

6. **Arm Playground** — choose A0, A1, A4; show the three answers side by side,
   retrieved chunks, latency and gold-source highlights.

7. **Assistant** — select a recorded question, click **Use this recorded
   question**, and open **How this answer was made**: guardrail, router,
   retrieval calls, chunks, citations, validator status, stage latency and
   token/cost line. Mention that live mode is the same UI with real calls.

8. **Methodology** — close on the protocol, limitations and reproduction
   commands.
