# vibe-agentic-ai-template

> Starter template for agentic AI apps using **LangGraph** + **CrewAI** with **Claude Sonnet**.

A production-ready Python project template that demonstrates the key patterns
for building reliable, multi-step AI agents — ready to fork and extend.

---

## Features

| Feature | Details |
|---|---|
| **LangGraph weather agent** | ReAct loop: agent → ToolNode → formatter; `add_messages` reducer; conditional routing |
| **LangGraph code-gen agent** | 5-node pipeline: planner → coder (tools) → reviewer → formatter; `with_structured_output` |
| **CrewAI code-gen crew** | 3-agent crew: Architect → Developer (`allow_code_execution=True`) → Reviewer |
| **Structured output** | Pydantic models extracted via a dedicated formatter node (avoids tool+schema conflicts) |
| **CLI** | Typer CLI with `--mock` mode — runs without API keys |
| **Tests** | pytest suite that passes with no API key (mock mode + structural tests) |
| **Ruff** | Pre-configured linting + formatting |
| **Modern packaging** | `pyproject.toml` with Hatchling, Python ≥ 3.11 |

---

## Architecture

### LangGraph Weather Graph

```
START → agent ──(tool calls?)──▶ tools → agent
                       │
                       └─(no tool calls)─▶ formatter → END
```

- **agent node** — Claude + `bind_tools([get_weather])` decides when to call the weather API
- **tools node** — `langgraph.prebuilt.ToolNode` executes tool calls
- **formatter node** — `with_structured_output(WeatherReport)` extracts typed output

### LangGraph Code-Generation Graph

```
START → planner → coder ──(tool calls?)──▶ tools → coder
                              │
                              └─(no tools)─▶ reviewer → formatter → END
```

- **planner** — creates an implementation plan (no tools)
- **coder** — writes code, can call `format_python_code` / `search_python_docs`
- **reviewer** — static code review pass
- **formatter** — structured `CodeArtifact` output

### CrewAI Code-Generation Crew

```
Architect ──(plan)──▶ Developer ──(code)──▶ Reviewer
```

- **Architect** — implementation plan (`allow_code_execution=False`)
- **Developer** — Python code with execution (`allow_code_execution=True`)
- **Reviewer** — quality review and final output
- Tasks are chained via `task.context` for automatic context passing

---

## Repository Structure

```
vibe-agentic-ai-template/
├── src/
│   └── vibe_agents/
│       ├── __init__.py
│       ├── cli.py                  # Typer CLI entry-point
│       ├── graphs/
│       │   ├── weather_graph.py    # LangGraph weather agent
│       │   └── code_graph.py       # LangGraph code-gen agent
│       ├── crews/
│       │   └── code_crew.py        # CrewAI code-gen crew
│       ├── tools/
│       │   ├── weather.py          # get_weather @tool (Open-Meteo)
│       │   └── code_tools.py       # format_python_code, search_python_docs
│       └── utils/
│           ├── config.py           # Env var helpers
│           ├── console.py          # Rich output helpers
│           └── mock_data.py        # Static fixtures for --mock mode
├── tests/
│   ├── test_config.py
│   ├── test_mock_data.py
│   ├── test_tools.py
│   ├── test_cli.py
│   ├── test_graph_structure.py
│   └── test_crew_structure.py
├── .env.example
├── .gitignore
├── LICENSE
├── Makefile
└── pyproject.toml
```

---

## Setup

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com) (only needed for non-mock mode)

### Install

```bash
# 1. Clone the template
git clone https://github.com/your-org/vibe-agentic-ai-template
cd vibe-agentic-ai-template

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install (dev mode)
pip install -e ".[dev]"
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (non-mock) | — | Your Anthropic API key |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-5` | Claude model to use |
| `MOCK_MODE` | No | `false` | Set `true` to skip real API calls |
| `LANGCHAIN_TRACING_V2` | No | — | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key |

---

## Running Examples

### Mock Mode (no API key needed)

```bash
# Weather agent
vibe-agents langgraph-weather --city London --mock

# Code-generation agent (LangGraph)
vibe-agents langgraph-code --prompt "Write a binary search function" --mock

# Code-generation crew (CrewAI)
vibe-agents crewai-code --prompt "Write a prime number checker" --mock
```

Or via Makefile:

```bash
make run-weather   # LangGraph weather (mock)
make run-code      # LangGraph code-gen (mock)
make run-crew      # CrewAI code-gen (mock)
```

### Real Mode (requires ANTHROPIC_API_KEY)

```bash
# Set your key in .env, then:
vibe-agents langgraph-weather --city "New York"
vibe-agents langgraph-code --prompt "Write a Redis cache wrapper class"
vibe-agents crewai-code --prompt "Write a decorator that retries on exception"
```

---

## Tests

```bash
# Run all tests (no API key needed)
pytest

# Or via Makefile
make test
```

Tests cover:
- Config validation and env var reading
- Mock data fixtures
- Tool function correctness (offline)
- CLI command invocation in mock mode
- LangGraph state schema and Pydantic model validation
- CrewAI agent/task factory configuration

---

## Linting

```bash
make lint     # ruff check
make format   # ruff check --fix + ruff format
```

---

## Extending the Template

### Add a new LangGraph tool

1. Create a function in `src/vibe_agents/tools/` decorated with `@tool`
2. Add it to the `TOOLS` list in the relevant graph file
3. Claude will automatically learn to call it from its docstring

### Add a new graph node

1. Write a node function `def my_node(state: MyState) -> dict:`
2. Register it: `builder.add_node("my_node", my_node)`
3. Wire it with `add_edge` or `add_conditional_edges`

### Add a new CrewAI agent

1. Write a factory function `make_my_agent(llm: LLM) -> Agent:` in `code_crew.py`
2. Create a corresponding `make_my_task()` factory
3. Add both to `build_code_crew()`

### Enable LangSmith tracing

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key
export LANGCHAIN_PROJECT=my-project
```

### Add memory / checkpointing (LangGraph)

```python
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
# Then invoke with a thread_id for multi-turn conversations
graph.invoke(state, config={"configurable": {"thread_id": "user-123"}})
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `EnvironmentError: ANTHROPIC_API_KEY is not set` | Add key to `.env` or use `--mock` |
| `ModuleNotFoundError: No module named 'vibe_agents'` | Run `pip install -e .` from the repo root |
| CrewAI `LLM` import error | Upgrade: `pip install --upgrade crewai` |
| LangGraph `add_messages` import error | Upgrade: `pip install --upgrade langgraph` |
| Anthropic tool+schema conflict | Use the two-step formatter node pattern (already implemented) |
| Open-Meteo timeout in tests | Tests use mock data — network errors fall back automatically |

---

## License

MIT — see [LICENSE](LICENSE).
