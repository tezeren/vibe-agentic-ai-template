"""
CrewAI Code-Generation Crew
============================
Demonstrates a multi-agent CrewAI workflow with Claude Sonnet via the
CrewAI LLM helper.

Agents
------
- **architect**   — plans the solution, no code execution needed
- **developer**   — writes Python code with allow_code_execution=True
- **reviewer**    — reviews code quality and produces a final report

Tasks flow: plan → implement → review

Key patterns demonstrated
-------------------------
- CrewAI LLM configuration for Claude (provider="anthropic")
- allow_code_execution=True on the developer agent
- Task context chaining (task.context = [previous_task])
- Crew.kickoff() with named inputs
- Keeping external calls optional / mockable (see run_code_crew)

Customise
---------
- Add a FileWriterTool to write the generated code to disk
- Use SerperDevTool for web search during research tasks
- Enable memory=True on the Crew for multi-turn retention
- Add a "tester" agent that runs the generated code with pytest
"""

from __future__ import annotations

from crewai import LLM, Agent, Crew, Task

from vibe_agents.utils.config import get_anthropic_api_key, get_model_name

# Type alias: factory functions accept either a real LLM object or a model string.
# Passing a string is useful in tests to avoid real API calls.
LLMInput = "LLM | str"

# ── 1. LLM factory ────────────────────────────────────────────────────────────


def _make_claude_llm() -> LLM:
    """
    Build a CrewAI LLM object pointing at Claude via the Anthropic provider.

    CrewAI's LLM class uses LiteLLM under the hood, so the model string
    must be prefixed with "anthropic/".
    """
    return LLM(
        model=f"anthropic/{get_model_name()}",
        api_key=get_anthropic_api_key(),
        temperature=0.2,
    )


# ── 2. Agent factory functions ────────────────────────────────────────────────


def make_architect_agent(llm: LLM | str) -> Agent:
    """
    Architect agent: analyses requirements and produces an implementation plan.

    No code execution — this agent is a pure reasoner.
    Customise the backstory to reflect your domain (e.g. "data engineering",
    "web development", "machine learning").
    """
    return Agent(
        role="Senior Software Architect",
        goal="Analyse the coding request and produce a clear, concise implementation plan",
        backstory=(
            "You are a senior software architect with 15 years of experience "
            "designing Python systems. You excel at breaking down complex problems "
            "into manageable, well-structured implementation plans."
        ),
        llm=llm,
        verbose=False,
        allow_code_execution=False,
        # Customise: add tools like SerperDevTool for research tasks
    )


def make_developer_agent(llm: LLM | str) -> Agent:
    """
    Developer agent: writes Python code and can execute it to verify correctness.

    allow_code_execution=True enables the internal CodeInterpreterTool.
    The agent will write code, run it, observe the output, and iterate.

    WARNING: code execution runs in the current Python environment.
    In production, use a sandboxed execution environment.
    """
    return Agent(
        role="Senior Python Developer",
        goal="Write clean, well-tested Python code following the architect's plan",
        backstory=(
            "You are a senior Python developer with expertise in software architecture "
            "and best practices. You write readable, well-documented code and always "
            "verify it works before submitting."
        ),
        llm=llm,
        verbose=False,
        allow_code_execution=True,  # enables CodeInterpreterTool
        # Customise: add FileWriterTool, FileReadTool, etc.
    )


def make_reviewer_agent(llm: LLM | str) -> Agent:
    """
    Reviewer agent: critiques code quality, correctness, and best practices.

    No code execution — focuses on static review.
    """
    return Agent(
        role="Code Reviewer",
        goal="Review the generated code for quality, correctness, and best practices",
        backstory=(
            "You are a meticulous code reviewer who cares deeply about code quality. "
            "You check for edge cases, security issues, performance problems, and "
            "ensure the code follows PEP 8 and modern Python conventions."
        ),
        llm=llm,
        verbose=False,
        allow_code_execution=False,
    )


# ── 3. Task factory functions ─────────────────────────────────────────────────


def make_plan_task(agent: Agent, coding_prompt: str) -> Task:
    """Create the planning task."""
    return Task(
        description=(
            f"Analyse this coding request and create a numbered implementation plan:\n\n"
            f"REQUEST: {coding_prompt}\n\n"
            "Your plan should cover:\n"
            "1. Required data structures\n"
            "2. Core algorithm / logic steps\n"
            "3. Error handling\n"
            "4. Any standard library modules to use\n"
            "Provide 4-6 clear steps."
        ),
        expected_output=(
            "A numbered implementation plan (4-6 steps) with no code yet — "
            "just clear descriptions of what needs to be done."
        ),
        agent=agent,
    )


def make_implement_task(agent: Agent, plan_task: Task, coding_prompt: str) -> Task:
    """Create the implementation task, depending on the plan task."""
    return Task(
        description=(
            f"Implement the following in Python:\n\n"
            f"REQUEST: {coding_prompt}\n\n"
            "Follow the architect's plan. Write complete, runnable Python code with:\n"
            "- A module docstring\n"
            "- Type hints on all functions\n"
            "- Docstrings on all public functions\n"
            "- A simple __main__ block demonstrating usage\n"
            "Execute the code to verify it runs without errors."
        ),
        expected_output=(
            "Complete Python source code inside a ```python ... ``` fenced block, "
            "followed by confirmation that it executed successfully."
        ),
        agent=agent,
        context=[plan_task],  # receives the plan as context
    )


def make_review_task(agent: Agent, implement_task: Task) -> Task:
    """Create the code review task, depending on the implementation task."""
    return Task(
        description=(
            "Review the Python code from the developer. Check for:\n"
            "1. Correctness and edge cases\n"
            "2. PEP 8 / PEP 257 compliance\n"
            "3. Appropriate use of type hints\n"
            "4. Missing error handling\n"
            "5. Any security or performance concerns\n"
            "Provide the final, improved code and a brief review summary."
        ),
        expected_output=(
            "A review summary (bullet points) followed by the final improved Python code "
            "in a ```python ... ``` fenced block."
        ),
        agent=agent,
        context=[implement_task],  # receives the implementation as context
    )


# ── 4. Crew builder ───────────────────────────────────────────────────────────


def build_code_crew() -> Crew:
    """
    Assemble the full code-generation Crew.

    Returns:
        A CrewAI Crew ready to kickoff().
    """
    llm = _make_claude_llm()

    architect = make_architect_agent(llm)
    developer = make_developer_agent(llm)
    reviewer = make_reviewer_agent(llm)

    plan_task = make_plan_task(architect, "{coding_prompt}")
    implement_task = make_implement_task(developer, plan_task, "{coding_prompt}")
    review_task = make_review_task(reviewer, implement_task)

    return Crew(
        agents=[architect, developer, reviewer],
        tasks=[plan_task, implement_task, review_task],
        verbose=False,
        # Customise: set memory=True to enable cross-task memory retention
        # memory=True,
    )


# ── 5. Convenience runner ─────────────────────────────────────────────────────


def run_code_crew(coding_prompt: str) -> str:
    """
    Run the code-generation Crew for *coding_prompt* and return the final output.

    Args:
        coding_prompt: Plain-English description of what to code.

    Returns:
        The final crew output as a string (reviewer's report + code).

    Raises:
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    crew = build_code_crew()
    result = crew.kickoff(inputs={"coding_prompt": coding_prompt})
    return str(result)
