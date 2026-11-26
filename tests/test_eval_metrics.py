"""Offline tests for the evaluation inputs and metric selection."""

from deepeval.test_case import ToolCall
from langchain_core.messages import AIMessage, HumanMessage

from src.core.graph import DeepSeekJudge, build_test_case, select_metrics


def test_no_gold_list_means_no_expected_tools():
    state = {
        "messages": [HumanMessage(content="question"), AIMessage(content="answer")],
    }

    test_case = build_test_case(state)

    assert test_case.expected_tools is None


def test_gold_list_sets_expected_tools():
    state = {
        "messages": [HumanMessage(content="draw"), AIMessage(content="drawn")],
        "expected_tools": ["spawn_geometry"],
    }

    test_case = build_test_case(state)

    assert test_case.expected_tools == [ToolCall(name="spawn_geometry")]


def test_tool_correctness_is_only_selected_with_a_gold_list():
    judge = DeepSeekJudge()
    without_gold = [type(metric).__name__ for metric in select_metrics(judge, False)]
    with_gold = [type(metric).__name__ for metric in select_metrics(judge, True)]

    assert "ToolCorrectnessMetric" not in without_gold
    assert with_gold[0] == "ToolCorrectnessMetric"
