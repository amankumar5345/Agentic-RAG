"""Schemas, file IO, targets and split assignment for LithoBench."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, model_validator

DATA_DIR = Path(__file__).resolve().parent / "data"

ROLE_NAMES = ("Safety Agent", "Maintainance Agent", "Draw Agent", "Chatbot")

QA_TARGETS = {
    "single_hop": 60,
    "multi_hop": 40,
    "numeric": 25,
    "hazard": 25,
    "unanswerable": 30,
}

ADVERSARIAL_TARGETS = {
    "jailbreak": 15,
    "off_topic": 10,
    "harmful_procedure": 10,
    "injection_direct": 10,
    "gibberish": 5,
    "benign_control": 30,
}

GEOMETRY_TARGET = 50
ROUTER_TARGETS = {role: 15 for role in ROLE_NAMES}

TARGETS = {
    "qa": QA_TARGETS,
    "adversarial": ADVERSARIAL_TARGETS,
    "geometry": GEOMETRY_TARGET,
    "router": ROUTER_TARGETS,
}

CASE_FILES = {
    "qa": "qa.jsonl",
    "adversarial": "adversarial.jsonl",
    "geometry": "geometry.jsonl",
    "router": "router.jsonl",
}


class GoldSource(BaseModel):
    doc: str
    section: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text_hash: str

    @model_validator(mode="after")
    def _check_offsets(self) -> "GoldSource":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class CaseBase(BaseModel):
    id: str
    type: str
    split: Literal["dev", "test"]
    origin: Literal["human", "llm_draft", "llm_edited"]
    status: Literal["draft", "verified"]
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _verified_needs_review(self) -> "CaseBase":
        if self.status == "verified" and (not self.reviewer or self.reviewed_at is None):
            raise ValueError("verified cases require reviewer and reviewed_at")
        return self


class QACase(CaseBase):
    type: Literal["qa"] = "qa"
    question: str
    gold_answer: str
    gold_sources: list[GoldSource]
    subtype: Literal["single_hop", "multi_hop", "numeric", "hazard", "unanswerable"]
    numeric_tolerance: Optional[float] = None

    @model_validator(mode="after")
    def _numeric_needs_tolerance(self) -> "QACase":
        if self.subtype == "numeric" and self.numeric_tolerance is None:
            raise ValueError("numeric cases require numeric_tolerance")
        return self


class AdversarialCase(CaseBase):
    type: Literal["adversarial"] = "adversarial"
    prompt: str
    category: Literal[
        "jailbreak",
        "off_topic",
        "harmful_procedure",
        "injection_direct",
        "gibberish",
        "benign_control",
    ]
    expected: Literal["block", "answer"]


class GeometryCase(CaseBase):
    type: Literal["geometry"] = "geometry"
    utterance: str
    gold_commands: list[dict]
    tolerance: float = 1.0
    expect_clarification: bool
    category: Literal["single", "multi_step", "loop", "synonym", "angle_words"]

    @model_validator(mode="after")
    def _clarification_matches_commands(self) -> "GeometryCase":
        missing = any(command.get("size") is None for command in self.gold_commands)
        if self.expect_clarification != missing:
            raise ValueError("expect_clarification must match whether any command lacks size")
        return self


class RouterCase(CaseBase):
    type: Literal["router"] = "router"
    query: str
    gold_role: Literal["Safety Agent", "Maintainance Agent", "Draw Agent", "Chatbot"]


Case = Annotated[
    Union[QACase, AdversarialCase, GeometryCase, RouterCase],
    Field(discriminator="type"),
]

CASE_ADAPTER = TypeAdapter(Case)


def source_key(source: GoldSource) -> tuple[str, int, int]:
    return (source.doc, source.char_start, source.char_end)


def case_path(case_type: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / CASE_FILES[case_type]


def read_cases(data_dir: Path = DATA_DIR) -> list[Case]:
    """Read every case, raising ValueError on the first malformed line."""
    cases: list[Case] = []
    for case_type, filename in CASE_FILES.items():
        path = data_dir / filename
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                cases.append(CASE_ADAPTER.validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return cases


def load_cases(include_draft: bool = False, data_dir: Path = DATA_DIR) -> list[Case]:
    """Evaluation loader: verified only unless include_draft (smoke tests)."""
    return [
        case
        for case in read_cases(data_dir)
        if include_draft or case.status == "verified"
    ]


def write_cases(cases: list[Case], data_dir: Path = DATA_DIR) -> None:
    duplicates = len(cases) - len({case.id for case in cases})
    if duplicates:
        raise ValueError(f"refusing to write {duplicates} duplicate case id(s)")
    data_dir.mkdir(parents=True, exist_ok=True)
    by_type: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        by_type[case.type].append(case)

    for case_type, filename in CASE_FILES.items():
        path = data_dir / filename
        rows = sorted(by_type.get(case_type, []), key=lambda case: case.id)
        payload = "".join(case.model_dump_json() + "\n" for case in rows)
        temp = path.with_name(f".{filename}.tmp")
        temp.write_text(payload, encoding="utf-8")
        temp.replace(path)


def _assign_qa_splits(qa_cases: list[QACase], rng: random.Random) -> dict[str, str]:
    parent: dict[tuple[str, int, int], tuple[str, int, int]] = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left, right):
        parent[find(left)] = find(right)

    keys_by_case: list[list[tuple[str, int, int]]] = []
    for case in qa_cases:
        keys = sorted({source_key(source) for source in case.gold_sources})
        if not keys:
            keys = [(f"__case__:{case.id}", 0, 0)]
        for key in keys:
            parent.setdefault(key, key)
        for key in keys[1:]:
            union(keys[0], key)
        keys_by_case.append(keys)

    groups: dict[tuple[str, int, int], list[QACase]] = defaultdict(list)
    for case, keys in zip(qa_cases, keys_by_case, strict=True):
        groups[find(keys[0])].append(case)

    group_list = list(groups.values())
    rng.shuffle(group_list)
    total = len(qa_cases)
    dev_target = round(0.3 * total)
    if total > 1:
        dev_target = min(max(dev_target, 1), total - 1)

    result: dict[str, str] = {}
    assigned = 0
    for group in group_list:
        if assigned + len(group) <= dev_target:
            split = "dev"
        elif assigned == 0 and len(group_list) == 1:
            split = "dev"
        else:
            split = "test"
        if split == "dev":
            assigned += len(group)
        for case in group:
            result[case.id] = split

    if qa_cases and not any(split == "dev" for split in result.values()):
        smallest = min(group_list, key=len)
        for case in smallest:
            result[case.id] = "dev"
    return result


def assign_splits(cases: list[Case], seed: int = 13) -> list[Case]:
    """Assign 30% dev / 70% test, keeping QA passages on one side."""
    rng = random.Random(seed)
    qa_cases = [case for case in cases if isinstance(case, QACase)]
    others = [case for case in cases if not isinstance(case, QACase)]

    split_by_id: dict[str, str] = {}
    if qa_cases:
        split_by_id.update(_assign_qa_splits(qa_cases, rng))

    rng.shuffle(others)
    dev_target = round(0.3 * len(others))
    for index, case in enumerate(others):
        split_by_id[case.id] = "dev" if index < dev_target else "test"

    return [case.model_copy(update={"split": split_by_id[case.id]}) for case in cases]


def make_id(prefix: str, payload: str) -> str:
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:10]}"
