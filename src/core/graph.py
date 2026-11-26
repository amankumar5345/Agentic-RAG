import os
import re
import warnings
from dataclasses import dataclass
from typing import Annotated, Callable, Literal, Optional, TypedDict

import litellm
import nest_asyncio
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    ToolCorrectnessMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, ToolCall
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langsmith import traceable
from nemoguardrails import LLMRails, RailsConfig
from pydantic import BaseModel, Field

from src.core.retriever import load_prompt
from src.core.tools import spawn_geometry

##-------------set llms & guardrails------------------
load_dotenv(override=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
nest_asyncio.apply()

# Only the config is loaded once (thread-safe).
# LLMRails itself is intentionally NOT instantiated here as a global —
# it binds to the thread's asyncio event loop at creation time, causing a
# deadlock on Streamlit's second re-run (which runs in a new thread).
config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "guardrails", "config"))
rails_config = RailsConfig.from_path(config_dir)


@dataclass(frozen=True)
class GraphConfig:
    """Model names and limits, so tests and callers can override them."""

    orchestrator_model: str = "openai/gpt-oss-120b"
    router_model: str = "openai/gpt-oss-20b"
    rails_model: str = "openai/gpt-oss-120b"
    judge_model: str = "deepseek/deepseek-chat"
    temperature: float = 0.0
    rails_temperature: float = 0.0
    max_rag_calls: int = 3


@dataclass(frozen=True)
class GuardrailDecision:
    blocked: bool
    message: str


def default_rails(config: GraphConfig | None = None) -> LLMRails:
    config = config or GraphConfig()
    thread_llm = ChatGroq(
        model=config.rails_model,
        temperature=config.rails_temperature,
        reasoning_format="hidden",
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    return LLMRails(rails_config, llm=thread_llm)


def run_guardrail(
    user_input: str,
    rails_provider=None,
    config: GraphConfig | None = None,
) -> GuardrailDecision:
    """Run the guardrail alone and return its decision.

    ``rails_provider`` may be an ``LLMRails`` or a zero-argument factory. A
    fresh instance is created per call for the Streamlit thread-loop reason.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)

    provider = rails_provider or default_rails
    thread_rails = provider() if callable(provider) else provider
    response = thread_rails.generate(messages=[{"role": "user", "content": user_input}])
    raw_content = (
        response.get("content", str(response))
        if isinstance(response, dict)
        else str(response)
    )
    clean_content = re.sub(
        r"<think>.*?</think>", "", raw_content, flags=re.DOTALL
    ).strip()

    if "QUERY_PASSED" in clean_content:
        return GuardrailDecision(blocked=False, message="")

    bot_message = clean_content
    if bot_message.startswith("DIALOG:"):
        bot_message = bot_message[len("DIALOG:"):].strip()
    elif bot_message.startswith("DIALOG"):
        bot_message = bot_message[len("DIALOG"):].strip()
    else:
        bot_message = re.sub(r"^\[RAIL_BLOCKED[^\]]*\]\s*", "", bot_message).strip()

    return GuardrailDecision(blocked=True, message=bot_message)


class DeepSeekJudge(DeepEvalBaseLLM):
    """Custom DeepEval judge using DeepSeek API.

    DeepSeek supports json_object mode but rejects json_schema (Pydantic strict mode)
    which is what DeepEval uses internally. This class:
      1. Forces response_format=json_object instead of json_schema.
      2. Injects the word 'json' into every prompt that uses JSON mode —
         a DeepSeek API requirement without which it rejects the request.
    DeepEval's metric parsers only care about receiving valid JSON, not how it was requested.
    """

    def __init__(self, model_name: str = "deepseek/deepseek-chat") -> None:
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    @traceable(run_type="llm", name="DeepSeek_Judge_Sync")
    def generate(self, prompt: str, schema=None):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        }
        if schema is not None:
            # Force json_object — DeepSeek rejects json_schema
            kwargs["response_format"] = {"type": "json_object"}
            # DeepSeek REQUIRES the word 'json' in the prompt to activate JSON mode
            if "json" not in prompt.lower():
                kwargs["messages"] = [{"role": "user", "content": prompt + "\n\nRespond with a JSON object."}]
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content
        try:
            cost = litellm.completion_cost(response)
        except Exception:
            cost = 0.0
        return content, cost

    @traceable(run_type="llm", name="DeepSeek_Judge_Async")
    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


#--------------classes-----------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    role: str
    is_blocked: Optional[bool]
    eval_results: Optional[dict]
    rag_calls: Optional[int]
    expected_tools: Optional[list[str]]


class RoleMaker(BaseModel):
    """The LLM reads this schema and its descriptions to choose the next route."""
    role: Literal["Safety Agent", "Maintainance Agent", "Draw Agent", "Chatbot"] = Field(
        description="""Choose the best agent based on these strict guidelines:
        - 'Safety Agent': Select if the question is related to hazards, emergency stops, operating thresholds, or safety compliance.
        - 'Maintainance Agent': Select if the question is about hardware wear and tear, servicing, calibration, or maintenance schedules.
        - 'Draw Agent': Select if the user commands the system to draw, make, generate, or display something on the Unreal Engine interface (e.g., 'make a rectangle').
        - 'Chatbot': Select for greetings, casual conversation, or anything that doesn't fit the technical categories above.
        """
    )


tools_list = [load_prompt, spawn_geometry]


#---------------evaluation helpers------------

def build_test_case(state: ChatState) -> LLMTestCase:
    """Collect the messages into a DeepEval test case.

    ``expected_tools`` is only copied from the state when a gold list is
    supplied; otherwise it stays ``None`` so tool correctness is not scored
    against the model's own output.
    """
    messages = state["messages"]
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    user_query = human_msgs[-1].content if human_msgs else ""
    final_output = messages[-1].content or ""

    executed_tools: list[ToolCall] = []
    retrieval_contexts: list[str] = []

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                executed_tools.append(
                    ToolCall(name=tc["name"], input_parameters=tc["args"])
                )
        elif isinstance(msg, ToolMessage) and getattr(msg, "name", None) == "load_prompt":
            retrieval_contexts.append(msg.content)

    gold_tools = state.get("expected_tools")
    expected_tools = [ToolCall(name=name) for name in gold_tools] if gold_tools else None

    return LLMTestCase(
        input=user_query,
        actual_output=final_output,
        tools_called=executed_tools,
        expected_tools=expected_tools,
        retrieval_context=retrieval_contexts if retrieval_contexts else ["N/A"]
    )


def select_metrics(judge: DeepEvalBaseLLM, include_tool_correctness: bool) -> list:
    """Build the metric list; tool correctness only when gold tools exist."""
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=judge),
        AnswerRelevancyMetric(threshold=0.5, model=judge),
        ContextualRelevancyMetric(threshold=0.5, model=judge),
    ]
    if include_tool_correctness:
        metrics.insert(0, ToolCorrectnessMetric(threshold=0.5, model=judge))
    return metrics


def run_deepeval(state: ChatState, judge: DeepEvalBaseLLM) -> dict:
    """Run the DeepEval metrics synchronously and return state['eval_results']."""
    test_case = build_test_case(state)
    metrics = select_metrics(judge, bool(state.get("expected_tools")))

    scores: dict = {}
    for metric in metrics:
        try:
            metric.measure(test_case)
            scores[metric.__class__.__name__] = {
                "score": round(metric.score, 3),
                "passed": metric.is_successful(),
            }
        except Exception as e:
            scores[metric.__class__.__name__] = {
                "score": 0.0,
                "passed": False,
                "error": str(e)
            }

    return {'eval_results': scores}


#---------------llm factories------------

def default_llms(config: GraphConfig, tools: list | None = None) -> dict:
    """Build the production clients. Tests inject their own via build_graph."""
    active_tools = list(tools) if tools is not None else tools_list
    api_key = os.environ.get("GROQ_API_KEY")
    orchestrator = ChatGroq(
        model=config.orchestrator_model,
        temperature=config.temperature,
        api_key=api_key
    ).bind_tools(active_tools, tool_choice='auto')
    router = ChatGroq(
        model=config.router_model,
        temperature=config.temperature,
        api_key=api_key
    ).with_structured_output(RoleMaker)
    judge = DeepSeekJudge(config.judge_model)

    def rails() -> LLMRails:
        return default_rails(config)

    return {
        "orchestrator": orchestrator,
        "router": router,
        "judge": judge,
        "rails": rails,
    }


def make_guarded_tool_node(config: GraphConfig, tool_node: ToolNode) -> Callable[[ChatState], dict]:
    """Wrap the tool node with the configurable retrieval-call budget."""

    def guarded_tool_node(state: ChatState) -> dict:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        rag_calls = [
            call for call in tool_calls
            if call["name"] == "load_prompt"
        ]

        calls_used: int = state.get("rag_calls") or 0

        if rag_calls and calls_used >= config.max_rag_calls:
            return {
                "messages": [
                    ToolMessage(
                        content=(
                            "Maximum retrieval calls reached for this question. "
                            "Answer using the existing retrieved context."
                        ),
                        tool_call_id=rag_calls[0]["id"],
                        name="load_prompt",
                    )
                ],
                "rag_calls": calls_used + len(rag_calls),
            }

        result = tool_node.invoke(state)

        return {
            **result,
            "rag_calls": calls_used + len(rag_calls),
        }

    return guarded_tool_node


#---------------graph factory------------

def build_graph(
    config: GraphConfig | None = None,
    llms: dict | None = None,
    enable_guardrail: bool = True,
    enable_router: bool = True,
    tools: list | None = None,
) -> StateGraph:
    """Build the agent graph, optionally with injected LLM clients and tools.

    ``llms`` may contain ``orchestrator``, ``router``, ``judge`` and ``rails``
    (an ``LLMRails`` instance or a zero-argument factory). It may also contain
    ``evaluator``, a ``state -> dict`` callable used by the evaluation node;
    tests use it to keep the graph offline.

    ``enable_guardrail`` and ``enable_router`` support the A5/A6/A7 ablations;
    both default to True, which is the original graph. ``tools`` lets callers
    replace ``spawn_geometry`` with a capturing stub.
    """
    config = config or GraphConfig()
    provided = llms or {}
    active_tools = list(tools) if tools is not None else tools_list
    if all(key in provided for key in ("orchestrator", "router", "judge", "rails")):
        dependencies = dict(provided)
    else:
        dependencies = {**default_llms(config, active_tools), **provided}

    evaluator = dependencies.get("evaluator")
    if evaluator is None:
        judge = dependencies["judge"]

        def _run_default_evaluator(state: ChatState) -> dict:
            return run_deepeval(state, judge)

        evaluator = _run_default_evaluator

    def guardrail_node(state: ChatState) -> dict:
        """Evaluates the user's query against NeMo Guardrails before role decision or orchestration.
        - If QUERY_PASSED: marks is_blocked=False and proceeds to role_decide.
        - If DIALOG or [RAIL_BLOCKED]: marks is_blocked=True, creates an AIMessage with the response, and sets role='Chatbot'.
        """
        human_msgs = [m for m in state['messages'] if isinstance(m, HumanMessage)]
        user_input = human_msgs[-1].content if human_msgs else ""

        decision = run_guardrail(user_input, dependencies["rails"], config)
        if not decision.blocked:
            return {"is_blocked": False}

        return {
            "messages": [AIMessage(content=decision.message)],
            "role": "Chatbot",
            "is_blocked": True,
        }

    def role_decide(state: ChatState) -> dict:
        """Classifies the user query into one of the four agent roles."""
        llm_role = dependencies["router"]
        human_msgs = [m for m in state['messages'] if isinstance(m, HumanMessage)]
        user_input = human_msgs[-1].content if human_msgs else ""
        decision = llm_role.invoke(user_input)
        return {'role': decision.role}

    def orchestrator_node(state: ChatState) -> dict:
        """Core decision engine — answers directly or delegates to load_prompt / spawn_geometry tools."""
        response = dependencies["orchestrator"].invoke(state['messages'])
        return {'messages': [response]}

    def evaluate_node(state: ChatState) -> dict:
        """Runs the injected evaluator (DeepEval by default) and stores eval_results."""
        return evaluator(state)

    #-----------------conditional routings---------

    def route_guardrail(state: ChatState) -> str:
        """Routes past the guardrail, to role_decide or straight to orchestration."""
        if state.get("is_blocked"):
            return END
        return "role_decide" if enable_router else "orchestrator"

    def route_orchestrator(state: ChatState) -> str:
        """Routes to tools if orchestrator requested a tool call, otherwise directly to evaluate."""
        next_node = tools_condition(state)
        return 'tools' if next_node == 'tools' else 'evaluate'

    #-------------graph-------------------

    tool_node = ToolNode(active_tools)
    guarded_tool_node = make_guarded_tool_node(config, tool_node)

    graph = StateGraph(ChatState)

    graph.add_node('guardrail',        guardrail_node)
    graph.add_node('role_decide',      role_decide)
    graph.add_node('orchestrator',     orchestrator_node)
    graph.add_node('tools',            guarded_tool_node)
    graph.add_node('evaluate',         evaluate_node)

    graph.add_edge(START, 'guardrail' if enable_guardrail else ( 'role_decide' if enable_router else 'orchestrator'))
    if enable_guardrail:
        graph.add_conditional_edges(
            'guardrail',
            route_guardrail,
            {
                'role_decide': 'role_decide',
                'orchestrator': 'orchestrator',
                END: END
            }
        )
    if enable_router:
        graph.add_edge('role_decide', 'orchestrator')
    graph.add_conditional_edges(
        'orchestrator',
        route_orchestrator,
        {'tools': 'tools', 'evaluate': 'evaluate'}
    )
    graph.add_edge('tools', 'orchestrator')
    graph.add_edge('evaluate', END)

    return graph


graph = build_graph()
chain = graph.compile()
