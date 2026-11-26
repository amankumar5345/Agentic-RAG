# Answer correctness rubric

You are grading an answer produced by a semiconductor cleanroom /
photolithography assistant against a gold answer.

Rules:

- Judge the factual content, not style, length or verbosity.
- The candidate may paraphrase. It is correct if it states the same facts as
  the gold answer and does not contradict them.
- If the gold answer is a number, the numeric path is handled separately; this
  rubric is for text answers.
- If the candidate refuses, is empty, or answers a different question, it is
  incorrect.
- Do not use outside knowledge.

Return **only** a JSON object with exactly these keys:

```json
{"correct": true, "reason": "one short sentence"}
```

Question:
$question

Gold answer:
$gold_answer

Candidate answer:
$answer
