"""
LangGraph Weather Graph
=======================
A multi-node ReAct-style graph that:

  1. **agent** node  — Claude decides which tool(s) to call
  2. **tools** node  — executes tool calls (ToolNode from langgraph.prebuilt)
  3. **formatter** node — uses with_structured_output to extract a typed report

Architecture
------------
START → agent → (has tool calls?) → tools → agent → formatter → END
                      ↓ (no tool calls)
                  formatter → END

Key patterns demonstrated
-------------------------
- TypedDict state with add_messages reducer
- ChatAnthropic.bind_tools() for tool-augmented generation
- langgraph.prebuilt.ToolNode
- Conditional routing based on tool-call presence
- Structured output via a Pydantic model (separate formatter node avoids
  the tool+structured-output conflict with Anthropic models)
- Clean separation of concerns: agent ≠ formatter

Customise
---------
- Swap get_weather for your own tools in TOOLS list
- Add more nodes (e.g. a "reviewer" that re-checks the report)
- Persist state with langgraph.checkpoint.MemorySaver for multi-turn
"""

from typing import Annotated, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from vibe_agents.tools.weather import get_weather
from vibe_agents.utils.config import get_anthropic_api_key, get_model_name

# ── 1. State definition ──────────────────────────────────────────────────────


class WeatherState(TypedDict):
    """
    Graph state.

    - messages: full conversation history; add_messages reducer appends
      new messages rather than replacing the list.
    - city: target city extracted from the user query.
    - weather_report: final structured output (populated by formatter node).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    city: str
    weather_report: "WeatherReport | None"


# ── 2. Structured output schema ───────────────────────────────────────────────


class WeatherReport(BaseModel):
    """Structured weather report extracted by the formatter node."""

    city: str = Field(description="Name of the city")
    temperature_celsius: float = Field(description="Current temperature in Celsius")
    condition: str = Field(description="Brief weather condition, e.g. 'Sunny', 'Rainy'")
    humidity_percent: int = Field(description="Relative humidity as a percentage 0-100")
    wind_kmh: float = Field(description="Wind speed in km/h")
    summary: str = Field(description="One-sentence plain-English weather summary")
    recommendation: str = Field(description="Practical tip for visitors, e.g. 'Bring an umbrella'")


# ── 3. Tools list ─────────────────────────────────────────────────────────────

# Add more tools here — Claude will decide when to call them.
TOOLS = [get_weather]

# ── 4. Node functions ─────────────────────────────────────────────────────────


def build_llm(*, structured_schema: type[BaseModel] | None = None) -> ChatAnthropic:
    """Build ChatAnthropic, optionally with structured output."""
    llm = ChatAnthropic(
        model=get_model_name(),
        api_key=get_anthropic_api_key(),
        temperature=0,
    )
    if structured_schema:
        return llm.with_structured_output(structured_schema)  # type: ignore[return-value]
    return llm


def agent_node(state: WeatherState) -> dict:
    """
    Agent node: Claude reads the conversation and calls tools as needed.

    The model is bound to our TOOLS list so it can emit tool-call messages.
    This node runs in a loop (via conditional_edges) until no more tool calls
    are requested.
    """
    # Bind tools so Claude can emit ToolCall messages
    llm_with_tools = ChatAnthropic(
        model=get_model_name(),
        api_key=get_anthropic_api_key(),
        temperature=0,
    ).bind_tools(TOOLS)

    # System prompt — customise to change agent behaviour
    system = SystemMessage(
        content=(
            "You are a helpful weather assistant. "
            "When the user asks about the weather in a city, call the get_weather tool. "
            "Report results clearly and concisely."
        )
    )

    response: AIMessage = llm_with_tools.invoke([system, *state["messages"]])  # type: ignore[assignment]
    return {"messages": [response]}


def formatter_node(state: WeatherState) -> dict:
    """
    Formatter node: uses with_structured_output to extract a typed WeatherReport.

    We use a *separate* node for structured output because Anthropic does not
    allow combining tool-use and forced structured schemas in the same request.
    By running this after the tool loop has finished we avoid that conflict.
    """
    llm_structured = ChatAnthropic(
        model=get_model_name(),
        api_key=get_anthropic_api_key(),
        temperature=0,
    ).with_structured_output(WeatherReport)

    # Build a prompt from the full conversation history
    history_text = "\n".join(
        f"{m.__class__.__name__}: {m.content}" for m in state["messages"]
    )
    prompt = (
        "Based on the following weather conversation, extract structured weather data.\n\n"
        f"{history_text}"
    )

    report: WeatherReport = llm_structured.invoke([HumanMessage(content=prompt)])  # type: ignore[assignment]
    return {"weather_report": report}


# ── 5. Routing logic ──────────────────────────────────────────────────────────


def should_use_tools(state: WeatherState) -> Literal["tools", "formatter"]:
    """
    Conditional edge: route to tool execution if the last AIMessage contains
    tool calls, otherwise proceed to the formatter.

    Customise: you could add a max-iteration guard here to avoid infinite loops.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "formatter"


# ── 6. Graph assembly ─────────────────────────────────────────────────────────


def build_weather_graph() -> StateGraph:
    """
    Build and compile the weather StateGraph.

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    tool_node = ToolNode(TOOLS)  # prebuilt node that executes tool calls

    builder = StateGraph(WeatherState)

    # Register nodes
    # Customise: add "reviewer", "summariser", or other nodes here
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("formatter", formatter_node)

    # Edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_use_tools)
    builder.add_edge("tools", "agent")  # loop back after tool execution
    builder.add_edge("formatter", END)

    return builder.compile()


# ── 7. Convenience runner ─────────────────────────────────────────────────────


def run_weather_query(city: str) -> WeatherReport:
    """
    Run the weather graph for *city* and return a structured WeatherReport.

    Args:
        city: The city to get weather for.

    Returns:
        A WeatherReport Pydantic model.

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    graph = build_weather_graph()
    user_message = HumanMessage(content=f"What is the weather like in {city} right now?")
    final_state = graph.invoke({"messages": [user_message], "city": city, "weather_report": None})
    report = final_state.get("weather_report")
    if report is None:
        raise RuntimeError("Weather graph did not produce a report.")
    return report
