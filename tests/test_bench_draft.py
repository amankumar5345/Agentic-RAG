"""Offline tests for drafting, parsing and the cached LLM client."""

import json
import re

import pytest

from bench.draft_cases import (
    Passage,
    PassageSampler,
    answer_absent_from_corpus,
    draft_adversarial,
    draft_geometry,
    draft_qa,
    draft_router,
    parse_json_object,
    split_corpus,
)
from litho_eval.client import LLMClient, assert_different_family

CORPUS = """--- Source: docs/a.pdf ---
## Intro
Laser safety requires goggles. The dose is 42 mJ.
## Process
The wafer is 300 mm wide.
--- Source: docs/b.pdf ---
## Cleanroom
Cleanroom class 100 requires gowning.
## Spin
Photoresist is spun at 3000 rpm.
"""


class FakeLLM:
    def __init__(self, responder):
        self.responder = responder
        self.calls = 0

    def complete(self, prompt):
        self.calls += 1
        return self.responder(prompt)


def qa_responder(prompt):
    if "multi-hop" in prompt:
        return json.dumps(
            {
                "question": "How do laser safety and gowning combine?",
                "gold_answer": "Goggles for laser work and gowning for the cleanroom.",
            }
        )
    if "numeric" in prompt:
        return json.dumps(
            {"question": "What is the dose?", "gold_answer": "42 mJ", "numeric_tolerance": 0.5}
        )
    if "hazard" in prompt:
        return json.dumps({"question": "Which PPE is required?", "gold_answer": "goggles"})
    if "NOT answered" in prompt:
        return json.dumps(
            {
                "question": "What is the maximum plutonium wafer diameter?",
                "gold_answer": "Not answerable from the provided corpus.",
            }
        )
    return json.dumps({"question": "What is the dose?", "gold_answer": "42 mJ"})


def test_split_corpus_keeps_offsets_and_docs():
    passages = split_corpus(CORPUS)

    assert {passage.doc for passage in passages} == {"a.pdf", "b.pdf"}
    for passage in passages:
        assert CORPUS[passage.char_start:passage.char_end] == passage.text
        assert passage.text_hash


def _passage(doc: str, index: int) -> Passage:
    return Passage(
        doc=doc,
        section="s",
        char_start=index * 10,
        char_end=index * 10 + 5,
        text=f"unique text {index}",
    )


def test_passage_sampler_draws_without_replacement():
    passages = [_passage("a.pdf" if index % 2 else "b.pdf", index) for index in range(6)]
    sampler = PassageSampler(passages, seed=1)

    first = sampler.take(3)
    second = sampler.take(3)

    assert len({passage.char_start for passage in first + second}) == 6


def test_passage_pairs_are_disjoint():
    passages = [_passage("a.pdf" if index % 2 else "b.pdf", index) for index in range(6)]

    pairs = PassageSampler(passages, seed=1).take_pairs(3)

    assert len(pairs) == 3
    used = [passage.char_start for pair in pairs for passage in pair]
    assert len(used) == len(set(used)) == 6
    assert all(first.doc != second.doc for first, second in pairs)


def test_adversarial_prompts_include_variation_numbers():
    prompts = []

    def responder(prompt):
        prompts.append(prompt)
        return json.dumps({"prompt": f"prompt {len(prompts)}"})

    draft_adversarial(FakeLLM(responder), limit=2, progress=lambda *_: None)

    assert "Variation 1" in prompts[0]
    assert "Variation 2" in prompts[1]


def test_parse_json_object_accepts_fenced_json():
    assert parse_json_object('```json\n{"question": "q"}\n```') == {"question": "q"}


def test_answer_absent_from_corpus():
    assert answer_absent_from_corpus("What is the plutonium wafer diameter?", CORPUS)
    assert not answer_absent_from_corpus("Does laser safety require goggles?", CORPUS)


def test_cached_llm_hits_disk_cache(tmp_path):
    calls = []

    def completer(messages):
        calls.append(messages)
        return "response"

    llm = LLMClient(
        "deepseek/deepseek-chat",
        cache_dir=tmp_path,
        mock=True,
        completer=completer,
    )

    assert llm.complete("same prompt").text == "response"
    assert llm.complete("same prompt").text == "response"
    assert llm.complete("other prompt").text == "response"

    assert len(calls) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_draft_model_must_differ_from_generator():
    with pytest.raises(ValueError):
        assert_different_family("openai/gpt-oss-20b", "openai/gpt-oss-120b")
    assert_different_family("deepseek/deepseek-chat", "openai/gpt-oss-120b")


def test_draft_qa_offline():
    llm = FakeLLM(qa_responder)
    texts = [
        "Laser safety requires goggles.",
        "The dose is 42 mJ.",
        "Cleanroom class 100 requires gowning.",
        "Photoresist is spun at 3000 rpm.",
        "The wafer is 300 mm wide.",
        "Wear PPE in the lab.",
        "Emergency stops are red.",
        "Laser alignment uses a 633 nm beam.",
        "The vacuum pump runs at 50 Hz.",
        "Hazard labels must be visible.",
    ]
    passages = [
        Passage(
            doc=f"d{index}.pdf",
            section="s",
            char_start=index * 200,
            char_end=index * 200 + len(text),
            text=text,
        )
        for index, text in enumerate(texts)
    ]

    cases = draft_qa(llm, passages, CORPUS, seed=1, limit=1, progress=lambda *_: None)

    assert len(cases) == 5
    assert {case.subtype for case in cases} == {
        "single_hop",
        "multi_hop",
        "numeric",
        "hazard",
        "unanswerable",
    }
    assert all(case.status == "draft" and case.origin == "llm_draft" for case in cases)

    multi_hop = next(case for case in cases if case.subtype == "multi_hop")
    assert len({(source.doc, source.char_start) for source in multi_hop.gold_sources}) == 2

    numeric = next(case for case in cases if case.subtype == "numeric")
    assert numeric.numeric_tolerance == 0.5


def test_draft_adversarial_offline():
    def responder(prompt):
        category = re.search(r'"(jailbreak|off_topic|harmful_procedure|injection_direct|gibberish|benign_control)"', prompt)
        return json.dumps({"prompt": f"prompt for {category.group(1)}"})

    cases = draft_adversarial(FakeLLM(responder), limit=1, progress=lambda *_: None)

    assert len(cases) == 6
    assert all(case.status == "draft" for case in cases)
    assert all(case.expected == ("answer" if case.category == "benign_control" else "block") for case in cases)


def test_draft_geometry_offline():
    def responder(prompt):
        category = re.search(r'in the "(\w+)" category', prompt).group(1)
        if category == "loop":
            return json.dumps(
                {
                    "utterance": "draw shrinking lines",
                    "gold_commands": [
                        {"shape": "line", "size": 10, "angle": 0},
                        {"shape": "line", "angle": 90},
                    ],
                    "tolerance": 1.0,
                }
            )
        return json.dumps(
            {
                "utterance": "draw a line of 10",
                "gold_commands": [{"shape": "line", "size": 10, "angle": 0}],
                "tolerance": 1.0,
            }
        )

    cases = draft_geometry(FakeLLM(responder), limit=1, progress=lambda *_: None)

    assert len(cases) == 5
    loop_case = next(case for case in cases if case.category == "loop")
    assert loop_case.expect_clarification is True
    assert all(
        case.expect_clarification is False for case in cases if case.category != "loop"
    )


def test_draft_router_offline():
    def responder(prompt):
        role = re.search(r'routed to the "(.+?)" agent', prompt).group(1)
        return json.dumps({"query": f"query for {role}"})

    cases = draft_router(FakeLLM(responder), limit=1, progress=lambda *_: None)

    assert len(cases) == 4
    assert {case.gold_role for case in cases} == {
        "Safety Agent",
        "Maintainance Agent",
        "Draw Agent",
        "Chatbot",
    }
