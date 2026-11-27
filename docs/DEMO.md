# 60-second demo script

1. Start the app:
   ```bash
   LITHO_MODE=demo streamlit run src/ui/app.py
   ```
   The DEMO banner appears; no keys, Ollama or Unreal needed.

2. **Benchmark** — show the page-level **Contains mock results, not measured**
   banner, the mock replay table, and one mock correctness bar. Point out that
   mock values are separated from measured values and never feed the headline
   takeaways or the README. A8 shows **NOT YET RUN** until a real run exists.

3. **Dataset Explorer** — filter to `qa / test / draft`, open one multi-hop
   case and show the gold source passage. Point out the **difficulty** column:
   stored values are shown as stored; unset single-hop cases show
   **derived (heuristic) medium**, multi-hop/numeric **hard**. Mention the
   dataset card at the bottom.

4. **Geometry Lab** — pick a case, show the JSON, validation result and 2D
   preview against the dashed gold outline. Press **Send to Unreal**; it writes
   the mailbox packet and shows a **SIMULATED** ack. The page banner marks the
   result as mock or simulated.

5. **Safety Lab** — pick a `jailbreak` sample, switch G0 → G3 and D0 → D3, and
   show the rule-based decision trace. Scroll to the poisoned-document pair to
   show the injection text and the sanitised context. The result tables carry a
   **MOCK** data column and the page banner.

6. **Arm Playground** — choose A0, A1, A4; show the three answers side by side,
   retrieved chunks, latency and gold-source highlights. A8 is not replayed
   from mock data.

7. **Assistant** — select a committed recorded question, click **Use this
   recorded question**, and open **How this answer was made**: schema version,
   guardrail, router role, rerank candidates, selected chunks, chunk ids,
   citation check, stage latency and token/cost line. The sample trace is
   labelled **MOCK** and **sample**; missing fields say **not recorded**.

8. **Methodology** — close on the protocol, limitations and reproduction
   commands.
