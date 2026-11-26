"""Offline graph tests with injected fake LLMs."""

from langchain_core.messages import AIMessage, HumanMessage

from src.core.graph import (
    GraphConfig,
    RoleMaker,
    build_graph,
    make_guarded_tool_node,
)


class FakeRails:
    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, messages):
        return {"content": self.content}


class FakeRouter:
    def __init__(self, role: str = "Chatbot") -> None:
        self.role = role
        self.calls = 0

    def invoke(self, _input):
        self.calls += 1
        return RoleMaker(role=self.role)


class FakeOrchestrator:
    def __init__(self, text: str = "final answer") -> None:
        self.text = text
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.text)


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, state):
        self.calls.append(state)
        return {"eval_results": {"stub": {"score": 1.0, "passed": True}}}


def make_dependencies(rails_content, evaluator=None, orchestrator=None):
    return {
        "orchestrator": orchestrator or FakeOrchestrator(),
        "router": FakeRouter(),
        "judge": None,
        "rails": FakeRails(rails_content),
        "evaluator": evaluator or FakeEvaluator(),
    }


def initial_state(text="hello"):
    return {
        "messages": [HumanMessage(content=text)],
        "role": "",
        "is_blocked": False,
        "eval_results": None,
        "rag_calls": 0,
    }


def test_build_graph_keeps_node_names():
    graph = build_graph(GraphConfig(), make_dependencies("QUERY_PASSED"))
    assert {"guardrail", "role_decide", "orchestrator", "tools", "evaluate"} <= set(graph.nodes)


def test_blocked_guardrail_goes_straight_to_end():
    evaluator = FakeEvaluator()
    orchestrator = FakeOrchestrator()
    dependencies = make_dependencies(
        "[RAIL_BLOCKED] Refused.", evaluator=evaluator, orchestrator=orchestrator
    )

    result = build_graph(GraphConfig(), dependencies).compile().invoke(initial_state())

    assert result["is_blocked"] is True
    assert result["role"] == "Chatbot"
    assert result["messages"][-1].content == "Refused."
    assert evaluator.calls == []
    assert orchestrator.calls == 0


def test_passed_guardrail_reaches_orchestrator_and_evaluator():
    evaluator = FakeEvaluator()
    orchestrator = FakeOrchestrator()
    dependencies = make_dependencies(
        "QUERY_PASSED", evaluator=evaluator, orchestrator=orchestrator
    )

    result = build_graph(GraphConfig(), dependencies).compile().invoke(initial_state())

    assert result["is_blocked"] is False
    assert result["role"] == "Chatbot"
    assert result["messages"][-1].content == "final answer"
    assert result["eval_results"] == {"stub": {"score": 1.0, "passed": True}}
    assert orchestrator.calls == 1
    assert len(evaluator.calls) == 1


def test_expected_tools_reach_the_evaluator():
    evaluator = FakeEvaluator()
    dependencies = make_dependencies("QUERY_PASSED", evaluator=evaluator)
    state = initial_state()
    state["expected_tools"] = ["spawn_geometry"]

    build_graph(GraphConfig(), dependencies).compile().invoke(state)

    assert evaluator.calls[0]["expected_tools"] == ["spawn_geometry"]


class StubToolNode:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, state):
        self.calls += 1
        return {"messages": []}


def tool_state(rag_calls):
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "load_prompt",
                        "args": {"user_input": "question"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        "rag_calls": rag_calls,
    }


def test_retrieval_budget_is_configurable():
    stub = StubToolNode()
    node = make_guarded_tool_node(GraphConfig(max_rag_calls=1), stub)

    result = node(tool_state(rag_calls=1))

    assert stub.calls == 0
    assert "Maximum retrieval calls" in result["messages"][0].content


def test_retrieval_allows_calls_under_the_budget():
    stub = StubToolNode()
    node = make_guarded_tool_node(GraphConfig(max_rag_calls=5), stub)

    result = node(tool_state(rag_calls=1))

    assert stub.calls == 1
    assert result["rag_calls"] == 2
