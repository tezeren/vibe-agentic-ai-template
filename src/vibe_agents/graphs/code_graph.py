"""
LangGraph Code-Generation Graph
================================
A multi-step agentic workflow that:

  1. **planner** node — Claude analyses the request and creates a plan
  2. **coder** node   — Claude writes Python code, optionally calling
                        format_python_code / search_python_docs tools
  3. **tools** node   — executes any tool calls from the coder
  4. **reviewer** node — Claude reviews and refines the generated code
  5. **formatter** node — extracts a typed CodeArtifact (structured output)

Architecture
------------
START → planner → coder → (has tool calls?) → tools → coder
                                ↓ (no tool calls)
                           reviewer → formatter → END

Key patterns demonstrated
-------------------------
- Multi-node pipeline with purpose-specific nodes
- Tool binding on the coder node only (planner/reviewer stay clean)
- Structured output on a separate formatter node
- Private "plan" field in state that isn't part of the input schema
- ToolNode from langgraph.prebuilt

Customise
---------
- Add a "tester" node that runs pytest on the generated code
- Add a "git_commit" node that pushes the code to a branch
- Replace search_python_docs with a vector store retriever
"""

from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from vibe_agents.tools.code_tools import format_python_code, search_python_docs
from vibe_agents.utils.config import get_anthropic_api_key, get_model_name

# ── 1. State ──────────────────────────────────────────────────────────────────


class CodeState(TypedDict):
    """
    Graph state for the code-generation workflow.

    - messages: full conversation (add_messages reducer)
    - prompt: original user request
    - plan: high-level plan produced by the planner node
    - code_artifact: final structured output
    """

    messages: Annotated[list[BaseMessage], add_messages]
    prompt: str
    plan: str
    code_artifact: "CodeArtifact | None"


# ── 2. Structured output schema ───────────────────────────────────────────────


class CodeArtifact(BaseModel):
    """Final structured code artefact emitted by the formatter node."""

    title: str = Field(description="Short title for the code snippet")
    language: str = Field(description="Programming language, e.g. 'python'")
    code: str = Field(description="The complete, runnable source code")
    explanation: str = Field(description="Plain-English explanation of what the code does")
    usage_example: str = Field(description="A short example showing how to call / run the code")
    dependencies: list[str] = Field(
        default_factory=list,
        description="Any pip packages required beyond the standard library",
    )


# ── 3. Tools ──────────────────────────────────────────────────────────────────

# Customise: add more tools here (e.g. file I/O, web search, linting)
CODE_TOOLS = [format_python_code, search_python_docs]


def _make_llm(*, bind_tools: bool = False) -> ChatAnthropic:
    llm = ChatAnthropic(
        model=get_model_name(),
        api_key=get_anthropic_api_key(),
        temperature=0.2,
    )
    if bind_tools:
        return llm.bind_tools(CODE_TOOLS)  # type: ignore[return-value]
    return llm


# ── 4. Node functions ─────────────────────────────────────────────────────────


def planner_node(state: CodeState) -> dict:
    """
    Planner node: creates a step-by-step implementation plan.

    The plan is stored in state["plan"] and included in subsequent prompts
    so the coder has clear guidance. No tools are needed here.
    """
    system = SystemMessage(
        content=(
            "You are a senior software architect. "
            "Given a coding request, produce a concise numbered implementation plan "
            "(3-6 steps). Do NOT write code yet — only the plan."
        )
    )
    user = HumanMessage(content=f"Coding request: {state['prompt']}")
    response: AIMessage = _make_llm().invoke([system, user])  # type: ignore[assignment]
    plan_text = str(response.content)
    return {
        "messages": [user, response],
        "plan": plan_text,
    }


def coder_node(state: CodeState) -> dict:
    """
    Coder node: writes Python code following the planner's plan.

    Tools are bound so the model can call format_python_code to verify
    syntax or search_python_docs for API details.
    """
    system = SystemMessage(
        content=(
            "You are an expert Python developer. "
            "Follow the implementation plan and write clean, well-documented Python code. "
            "Use the format_python_code tool to verify syntax if needed. "
            "Return only working Python code."
        )
    )
    plan_msg = HumanMessage(
        content=f"Implementation plan:\n{state['plan']}\n\nNow write the code."
    )
    response: AIMessage = _make_llm(bind_tools=True).invoke(  # type: ignore[assignment]
        [system, *state["messages"], plan_msg]
    )
    return {"messages": [plan_msg, response]}


def reviewer_node(state: CodeState) -> dict:
    """
    Reviewer node: critiques and refines the generated code.

    No tool access — purely analytical. The model reviews the last
    generated code message and emits improvements.
    """
    system = SystemMessage(
        content=(
            "You are a meticulous code reviewer. "
            "Review the code generated so far for correctness, readability, and edge cases. "
            "Provide the final improved version of the code."
        )
    )
    response: AIMessage = _make_llm().invoke([system, *state["messages"]])  # type: ignore[assignment]
    return {"messages": [response]}


def formatter_node(state: CodeState) -> dict:
    """
    Formatter node: extracts a typed CodeArtifact using with_structured_output.

    Runs after the tool loop completes so there is no tool+schema conflict.
    """
    llm_structured = ChatAnthropic(
        model=get_model_name(),
        api_key=get_anthropic_api_key(),
        temperature=0,
    ).with_structured_output(CodeArtifact)

    history_text = "\n".join(
        f"{m.__class__.__name__}: {m.content}" for m in state["messages"]
    )
    prompt = (
        "Extract structured information from the following code-generation conversation "
        "into a CodeArtifact.\n\n"
        f"{history_text}"
    )
    artifact: CodeArtifact = llm_structured.invoke([HumanMessage(content=prompt)])  # type: ignore[assignment]
    return {"code_artifact": artifact}


# ── 5. Routing ────────────────────────────────────────────────────────────────


def coder_should_use_tools(state: CodeState) -> Literal["tools", "reviewer"]:
    """Route to tools if the coder emitted tool calls, else go to reviewer."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "reviewer"


# ── 6. Graph assembly ─────────────────────────────────────────────────────────


def build_code_graph() -> StateGraph:
    """Build and compile the code-generation StateGraph."""
    tool_node = ToolNode(CODE_TOOLS)

    builder = StateGraph(CodeState)

    builder.add_node("planner", planner_node)
    builder.add_node("coder", coder_node)
    builder.add_node("tools", tool_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("formatter", formatter_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "coder")
    builder.add_conditional_edges("coder", coder_should_use_tools)
    builder.add_edge("tools", "coder")  # loop back after tool execution
    builder.add_edge("reviewer", "formatter")
    builder.add_edge("formatter", END)

    return builder.compile()


# ── 7. Convenience runner ─────────────────────────────────────────────────────


def run_code_generation(prompt: str) -> CodeArtifact:
    """
    Run the code-generation graph for *prompt* and return a CodeArtifact.

    Args:
        prompt: Plain-English description of the code to generate.

    Returns:
        A CodeArtifact Pydantic model with code, explanation, and usage example.
    """
    graph = build_code_graph()
    final_state = graph.invoke(
        {
            "messages": [],
            "prompt": prompt,
            "plan": "",
            "code_artifact": None,
        }
    )
    artifact = final_state.get("code_artifact")
    if artifact is None:
        raise RuntimeError("Code graph did not produce an artifact.")
    return artifact
