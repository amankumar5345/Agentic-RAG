import os
import re
import warnings
import nest_asyncio
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from nemoguardrails import LLMRails, RailsConfig

from src.core.tools import spawn_geometry
from src.core.retriever import load_prompt

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from deepeval.models.base_model import DeepEvalBaseLLM
import litellm
from langsmith import traceable

##-------------set llms & guardrails------------------
load_dotenv(override=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
nest_asyncio.apply()

api_key = os.environ.get("GROQ_API_KEY")

model = ChatGroq(
    model='openai/gpt-oss-120b',
    temperature=0.1,
    api_key=api_key
)

model_for_role = ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.1,
    api_key=api_key
)

# Only the config is loaded once (thread-safe).
# LLMRails itself is intentionally NOT instantiated here as a global —
# it binds to the thread's asyncio event loop at creation time, causing a
# deadlock on Streamlit's second re-run (which runs in a new thread).
config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "guardrails", "config"))
rails_config = RailsConfig.from_path(config_dir)

class DeepSeekJudge(DeepEvalBaseLLM):
    """Custom DeepEval judge using DeepSeek API.

    DeepSeek supports json_object mode but rejects json_schema (Pydantic strict mode)
    which is what DeepEval uses internally. This class:
      1. Forces response_format=json_object instead of json_schema.
      2. Injects the word 'json' into every prompt that uses JSON mode —
         a DeepSeek API requirement without which it rejects the request.
    DeepEval's metric parsers only care about receiving valid JSON, not how it was requested.
    """

    def load_model(self):
        return "deepseek/deepseek-chat"

    @traceable(run_type="llm", name="DeepSeek_Judge_Sync")
    def generate(self, prompt: str, schema=None):
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": "deepseek/deepseek-chat",
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
        return "deepseek/deepseek-chat"


deepseek_judge = DeepSeekJudge()

tools_list = [load_prompt, spawn_geometry]
orchestrator_llm = model.bind_tools(tools_list, tool_choice='auto')

#--------------classes-----------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    role: str
    is_blocked: Optional[bool]
    eval_results: Optional[dict]
    rag_calls: Optional[int]   


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


#---------------nodes------------

def guardrail_node(state: ChatState) -> dict:
    """Evaluates the user's query against NeMo Guardrails before role decision or orchestration.
    - If QUERY_PASSED: marks is_blocked=False and proceeds to role_decide.
    - If DIALOG or [RAIL_BLOCKED]: marks is_blocked=True, creates an AIMessage with the response, and sets role='Chatbot'.

    IMPORTANT: LLMRails is instantiated fresh on every call.
    This is intentional — NeMo Guardrails binds to the asyncio event loop of
    the thread that creates it. Streamlit re-runs each query in a new thread,
    so a global LLMRails singleton would try to post async work to the dead
    first-thread loop and hang forever (deadlock). Creating it per-call
    ensures it always binds to the current thread's own event loop.
    """
    import asyncio

    # Set up a fresh event loop for this thread if needed, then patch it
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("Loop is closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)

    
    thread_rail_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        reasoning_format="hidden",
        api_key=api_key
    )
    thread_rails = LLMRails(rails_config, llm=thread_rail_llm)

    human_msgs = [m for m in state['messages'] if isinstance(m, HumanMessage)]
    user_input = human_msgs[-1].content if human_msgs else ""

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
        return {"is_blocked": False}

    # Clean up prefixes for UI display
    bot_message = clean_content
    if bot_message.startswith("DIALOG:"):
        bot_message = bot_message[len("DIALOG:"):].strip()
    elif bot_message.startswith("DIALOG"):
        bot_message = bot_message[len("DIALOG"):].strip()
    else:
        bot_message = re.sub(r"^\[RAIL_BLOCKED[^\]]*\]\s*", "", bot_message).strip()

    return {
        "messages": [AIMessage(content=bot_message)],
        "role": "Chatbot",
        "is_blocked": True,
    }


def role_decide(state: ChatState) -> dict:
    """Classifies the user query into one of the four agent roles."""
    llm_role = model_for_role.with_structured_output(RoleMaker)
    human_msgs = [m for m in state['messages'] if isinstance(m, HumanMessage)]
    user_input = human_msgs[-1].content if human_msgs else ""
    decision = llm_role.invoke(user_input)
    return {'role': decision.role}


def orchestrator_node(state: ChatState) -> dict:
    """Core decision engine — answers directly or delegates to load_prompt / spawn_geometry tools."""
    response = orchestrator_llm.invoke(state['messages'])
    return {'messages': [response]}


def evaluate_node(state: ChatState) -> dict:
    """Runs DeepEval metrics synchronously.
    Scores are printed to the terminal and stored in state['eval_results']."""
    messages = state['messages']
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

    test_case = LLMTestCase(
        input=user_query,
        actual_output=final_output,
        tools_called=executed_tools,
        expected_tools=executed_tools,
        retrieval_context=retrieval_contexts if retrieval_contexts else ["N/A"]
    )

    metrics = [
        # ToolCorrectnessMetric(threshold=0.5, model=deepseek_judge),
        FaithfulnessMetric(threshold=0.5, model=deepseek_judge),
        AnswerRelevancyMetric(threshold=0.5, model=deepseek_judge),
        ContextualRelevancyMetric(threshold=0.5, model=deepseek_judge),
    ]

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


def guarded_tool_node(state: ChatState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    rag_calls = [
        call for call in tool_calls
        if call["name"] == "load_prompt"
    ]

    calls_used: int = state.get("rag_calls") or 0   # Fixed: safe fallback when key absent

    if rag_calls and calls_used >= 3:
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


#-----------------conditional routings---------

def route_guardrail(state: ChatState) -> str:
    """Routes to role_decide if guardrail passed, otherwise directly to END."""
    if state.get("is_blocked"):
        return END
    return "role_decide"


def route_orchestrator(state: ChatState) -> str:
    """Routes to tools if orchestrator requested a tool call, otherwise directly to evaluate."""
    next_node = tools_condition(state)
    return 'tools' if next_node == 'tools' else 'evaluate'


#-------------graph-------------------

tool_node = ToolNode(tools_list)

graph = StateGraph(ChatState)

graph.add_node('guardrail',        guardrail_node)
graph.add_node('role_decide',      role_decide)
graph.add_node('orchestrator',     orchestrator_node)
graph.add_node('tools',            guarded_tool_node)
graph.add_node('evaluate',         evaluate_node)

graph.add_edge(START, 'guardrail')
graph.add_conditional_edges(
    'guardrail',
    route_guardrail,
    {
        'role_decide': 'role_decide',
        END: END
    }
)
graph.add_edge('role_decide', 'orchestrator')
graph.add_conditional_edges(
    'orchestrator',
    route_orchestrator,
    {'tools': 'tools', 'evaluate': 'evaluate'}
)
graph.add_edge('tools', 'orchestrator')
graph.add_edge('evaluate', END)

chain = graph.compile()