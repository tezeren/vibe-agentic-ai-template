"""
Tests for the Typer CLI in vibe_agents.cli

Uses Typer's CliRunner to invoke commands in-process.
All tests use --mock so no API key is required.
"""

import pytest
from typer.testing import CliRunner

from vibe_agents.cli import app

runner = CliRunner()


class TestLanggraphWeatherCommand:
    def test_mock_mode_succeeds(self) -> None:
        result = runner.invoke(app, ["langgraph-weather", "--mock"])
        assert result.exit_code == 0

    def test_mock_mode_with_city(self) -> None:
        result = runner.invoke(app, ["langgraph-weather", "--city", "Tokyo", "--mock"])
        assert result.exit_code == 0

    def test_no_api_key_without_mock_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(app, ["langgraph-weather", "--city", "London"])
        assert result.exit_code != 0


class TestLanggraphCodeCommand:
    def test_mock_mode_succeeds(self) -> None:
        result = runner.invoke(app, ["langgraph-code", "--mock"])
        assert result.exit_code == 0

    def test_mock_mode_with_prompt(self) -> None:
        result = runner.invoke(
            app,
            ["langgraph-code", "--prompt", "Write a hello world function", "--mock"],
        )
        assert result.exit_code == 0

    def test_no_api_key_without_mock_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(app, ["langgraph-code"])
        assert result.exit_code != 0


class TestCrewaiCodeCommand:
    def test_mock_mode_succeeds(self) -> None:
        result = runner.invoke(app, ["crewai-code", "--mock"])
        assert result.exit_code == 0

    def test_mock_mode_with_prompt(self) -> None:
        result = runner.invoke(
            app,
            ["crewai-code", "--prompt", "Write a prime number checker", "--mock"],
        )
        assert result.exit_code == 0

    def test_no_api_key_without_mock_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = runner.invoke(app, ["crewai-code"])
        assert result.exit_code != 0


class TestHelpText:
    def test_top_level_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "langgraph-weather" in result.output
        assert "langgraph-code" in result.output
        assert "crewai-code" in result.output
