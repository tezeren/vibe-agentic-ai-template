"""
vibe-agentic-ai CLI
===================
Entry-point for the Typer command-line interface.

Commands
--------
  langgraph-weather   — Run the LangGraph weather agent
  langgraph-code      — Run the LangGraph code-generation agent
  crewai-code         — Run the CrewAI code-generation crew

All commands accept --mock to run without making real API calls, which is
useful for demos, onboarding, and CI environments without API keys.

Usage
-----
  # Install and run:
  pip install -e .
  vibe-agents --help

  # Or directly:
  python -m vibe_agents.cli langgraph-weather --city London --mock
"""

from __future__ import annotations

import typer
from rich.console import Console

from vibe_agents.utils.console import print_error, print_header, print_info, print_result

app = typer.Typer(
    name="vibe-agents",
    help="Agentic AI starter — LangGraph + CrewAI with Claude Sonnet",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ── Shared option types ───────────────────────────────────────────────────────

MockOption = typer.Option(False, "--mock", help="Run in mock mode (no API key needed)")


# ── Command: langgraph-weather ────────────────────────────────────────────────


@app.command(name="langgraph-weather")
def langgraph_weather(
    city: str = typer.Option("London", "--city", "-c", help="City to get weather for"),
    mock: bool = MockOption,
) -> None:
    """
    [bold cyan]LangGraph[/bold cyan] — multi-node weather agent.

    Runs a ReAct-style StateGraph: agent → tools → formatter.
    Uses Claude Sonnet with tool binding and structured output.

    Examples:

      vibe-agents langgraph-weather --city Tokyo

      vibe-agents langgraph-weather --city "New York" --mock
    """
    print_header(
        "LangGraph Weather Agent",
        f"City: {city} | Mock: {mock}",
    )

    if mock:
        print_info("Mock mode — returning static data (no API call)")
        _show_mock_weather(city)
        return

    # Real API call
    try:
        _require_api_key()
        from vibe_agents.graphs.weather_graph import run_weather_query

        with console.status("[cyan]Running weather graph…[/cyan]"):
            report = run_weather_query(city)

        print_result("Weather Report", _format_weather_report(report))
    except OSError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:  # noqa: BLE001
        print_error(f"Graph error: {e}")
        raise typer.Exit(code=1) from e


def _show_mock_weather(city: str) -> None:
    from vibe_agents.utils.mock_data import get_mock_weather

    data = get_mock_weather(city)
    print_result(
        "Mock Weather Report",
        f"City:        {data['city']}\n"
        f"Temperature: {data['temperature_celsius']}°C\n"
        f"Condition:   {data['condition']}\n"
        f"Humidity:    {data['humidity_percent']}%\n"
        f"Wind:        {data['wind_kmh']} km/h\n"
        f"Summary:     {data['summary']}",
    )


def _format_weather_report(report: object) -> str:
    """Format a WeatherReport Pydantic model as human-readable text."""
    r = report  # type: ignore[assignment]
    return (
        f"City:           {r.city}\n"  # type: ignore[attr-defined]
        f"Temperature:    {r.temperature_celsius}°C\n"  # type: ignore[attr-defined]
        f"Condition:      {r.condition}\n"  # type: ignore[attr-defined]
        f"Humidity:       {r.humidity_percent}%\n"  # type: ignore[attr-defined]
        f"Wind:           {r.wind_kmh} km/h\n"  # type: ignore[attr-defined]
        f"Summary:        {r.summary}\n"  # type: ignore[attr-defined]
        f"Recommendation: {r.recommendation}"  # type: ignore[attr-defined]
    )


# ── Command: langgraph-code ───────────────────────────────────────────────────


@app.command(name="langgraph-code")
def langgraph_code(
    prompt: str = typer.Option(
        "Write a Python function to calculate the Fibonacci sequence",
        "--prompt",
        "-p",
        help="Coding task description",
    ),
    mock: bool = MockOption,
) -> None:
    """
    [bold cyan]LangGraph[/bold cyan] — multi-node code-generation agent.

    Pipeline: planner → coder (with tools) → reviewer → formatter.
    Returns a structured CodeArtifact with code, explanation, and usage example.

    Examples:

      vibe-agents langgraph-code --prompt "Write a binary search function"

      vibe-agents langgraph-code --mock
    """
    print_header(
        "LangGraph Code-Generation Agent",
        f"Mock: {mock}",
    )
    console.print(f"[bold]Prompt:[/bold] {prompt}\n")

    if mock:
        print_info("Mock mode — returning static code (no API call)")
        from vibe_agents.utils.mock_data import MOCK_CODE_RESPONSE

        print_result("Generated Code (mock)", MOCK_CODE_RESPONSE, language="python")
        return

    try:
        _require_api_key()
        from vibe_agents.graphs.code_graph import run_code_generation

        with console.status("[cyan]Running code-generation graph…[/cyan]"):
            artifact = run_code_generation(prompt)

        print_result("Title", artifact.title)
        print_result("Code", artifact.code, language="python")
        print_result("Explanation", artifact.explanation)
        print_result("Usage Example", artifact.usage_example, language="python")
        if artifact.dependencies:
            print_result("Dependencies", "  ".join(artifact.dependencies))
    except OSError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:  # noqa: BLE001
        print_error(f"Graph error: {e}")
        raise typer.Exit(code=1) from e


# ── Command: crewai-code ──────────────────────────────────────────────────────


@app.command(name="crewai-code")
def crewai_code(
    prompt: str = typer.Option(
        "Write a Python function to check if a number is prime",
        "--prompt",
        "-p",
        help="Coding task description",
    ),
    mock: bool = MockOption,
) -> None:
    """
    [bold magenta]CrewAI[/bold magenta] — multi-agent code-generation crew.

    Three-agent pipeline: Architect → Developer (with code execution) → Reviewer.

    Examples:

      vibe-agents crewai-code --prompt "Write a stack data structure"

      vibe-agents crewai-code --mock
    """
    print_header(
        "CrewAI Code-Generation Crew",
        f"Mock: {mock}",
    )
    console.print(f"[bold]Prompt:[/bold] {prompt}\n")

    if mock:
        print_info("Mock mode — returning static code (no API call)")
        from vibe_agents.utils.mock_data import MOCK_CODE_RESPONSE, MOCK_CODE_REVIEW

        print_result("Code Review (mock)", MOCK_CODE_REVIEW)
        print_result("Generated Code (mock)", MOCK_CODE_RESPONSE, language="python")
        return

    try:
        _require_api_key()
        from vibe_agents.crews.code_crew import run_code_crew

        with console.status("[magenta]Running CrewAI crew…[/magenta]"):
            result = run_code_crew(prompt)

        print_result("Crew Output", result)
    except OSError as e:
        print_error(str(e))
        raise typer.Exit(code=1) from e
    except Exception as e:  # noqa: BLE001
        print_error(f"Crew error: {e}")
        raise typer.Exit(code=1) from e


# ── Helpers ───────────────────────────────────────────────────────────────────


def _require_api_key() -> None:
    """Raise EnvironmentError if the Anthropic API key is missing."""
    from vibe_agents.utils.config import get_anthropic_api_key

    get_anthropic_api_key()  # raises EnvironmentError if absent


# ── Entry-point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Module entry-point used by `python -m vibe_agents.cli`."""
    app()


if __name__ == "__main__":
    main()
