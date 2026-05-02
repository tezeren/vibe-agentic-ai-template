"""
Tests for CrewAI crew structure (no API calls).

These tests verify agent/task factory functions produce objects with the
correct configuration — without invoking Claude or executing code.

CrewAI's Pydantic models require real Agent/Task/LLM objects (not MagicMock),
so we construct minimal but real instances using a stub model string.
"""



# A dummy model string — CrewAI accepts a string as an LLM shorthand.
# This avoids any real API calls while satisfying Pydantic validation.
_STUB_LLM = "anthropic/claude-haiku-3-5"


class TestCrewAgentFactories:
    """Verify agent factory functions produce correctly configured Agent objects."""

    def test_architect_agent_has_correct_role(self) -> None:
        from vibe_agents.crews.code_crew import make_architect_agent

        agent = make_architect_agent(_STUB_LLM)
        assert "Architect" in agent.role or "architect" in agent.role.lower()

    def test_architect_agent_no_code_execution(self) -> None:
        from vibe_agents.crews.code_crew import make_architect_agent

        agent = make_architect_agent(_STUB_LLM)
        assert agent.allow_code_execution is False

    def test_developer_agent_has_code_execution(self) -> None:
        from vibe_agents.crews.code_crew import make_developer_agent

        agent = make_developer_agent(_STUB_LLM)
        assert agent.allow_code_execution is True

    def test_reviewer_agent_no_code_execution(self) -> None:
        from vibe_agents.crews.code_crew import make_reviewer_agent

        agent = make_reviewer_agent(_STUB_LLM)
        assert agent.allow_code_execution is False


class TestCrewTaskFactories:
    """Verify task factory functions wire context correctly."""

    def _make_agent(self, role: str = "Test Agent") -> object:
        """Build a minimal real CrewAI Agent for test purposes."""
        from crewai import Agent

        return Agent(
            role=role,
            goal="Test goal",
            backstory="Test backstory",
            llm=_STUB_LLM,
            verbose=False,
        )

    def test_plan_task_includes_prompt(self) -> None:
        from vibe_agents.crews.code_crew import make_plan_task

        agent = self._make_agent("Planner")
        task = make_plan_task(agent, "Write a sorting algorithm")  # type: ignore[arg-type]
        assert "sorting algorithm" in task.description

    def test_implement_task_has_context(self) -> None:
        from vibe_agents.crews.code_crew import make_implement_task, make_plan_task

        agent = self._make_agent("Developer")
        plan_task = make_plan_task(agent, "Write a sorting algorithm")  # type: ignore[arg-type]
        impl_task = make_implement_task(agent, plan_task, "Write a sorting algorithm")  # type: ignore[arg-type]
        assert plan_task in impl_task.context

    def test_review_task_has_context(self) -> None:
        from vibe_agents.crews.code_crew import (
            make_implement_task,
            make_plan_task,
            make_review_task,
        )

        agent = self._make_agent("Reviewer")
        plan_task = make_plan_task(agent, "Write a sorting algorithm")  # type: ignore[arg-type]
        impl_task = make_implement_task(agent, plan_task, "Write a sorting algorithm")  # type: ignore[arg-type]
        review_task = make_review_task(agent, impl_task)  # type: ignore[arg-type]
        assert impl_task in review_task.context
