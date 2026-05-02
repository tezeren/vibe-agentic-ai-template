"""
Tests for LangGraph graph structure (no API calls).

These tests verify that graphs compile correctly, state schemas are valid,
and Pydantic models have the expected fields — without invoking Claude.
"""

import pytest


class TestWeatherGraphStructure:
    def test_weather_state_has_required_fields(self) -> None:
        from vibe_agents.graphs.weather_graph import WeatherState

        # TypedDict introspection
        annotations = WeatherState.__annotations__
        assert "messages" in annotations
        assert "city" in annotations
        assert "weather_report" in annotations

    def test_weather_report_pydantic_fields(self) -> None:
        from vibe_agents.graphs.weather_graph import WeatherReport

        fields = WeatherReport.model_fields
        expected = {"city", "temperature_celsius", "condition", "humidity_percent", "wind_kmh", "summary", "recommendation"}
        assert expected.issubset(fields.keys())

    def test_weather_report_instantiation(self) -> None:
        from vibe_agents.graphs.weather_graph import WeatherReport

        report = WeatherReport(
            city="TestCity",
            temperature_celsius=15.0,
            condition="Sunny",
            humidity_percent=60,
            wind_kmh=10.0,
            summary="Nice day",
            recommendation="No jacket needed",
        )
        assert report.city == "TestCity"
        assert report.temperature_celsius == 15.0

    def test_graph_compiles_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Graph compilation should not require an API key."""
        # build_weather_graph only calls get_api_key inside nodes, not at compile time
        from vibe_agents.graphs.weather_graph import build_weather_graph

        graph = build_weather_graph()
        assert graph is not None


class TestCodeGraphStructure:
    def test_code_state_has_required_fields(self) -> None:
        from vibe_agents.graphs.code_graph import CodeState

        annotations = CodeState.__annotations__
        assert "messages" in annotations
        assert "prompt" in annotations
        assert "plan" in annotations
        assert "code_artifact" in annotations

    def test_code_artifact_pydantic_fields(self) -> None:
        from vibe_agents.graphs.code_graph import CodeArtifact

        fields = CodeArtifact.model_fields
        expected = {"title", "language", "code", "explanation", "usage_example", "dependencies"}
        assert expected.issubset(fields.keys())

    def test_code_artifact_instantiation(self) -> None:
        from vibe_agents.graphs.code_graph import CodeArtifact

        artifact = CodeArtifact(
            title="Hello World",
            language="python",
            code='print("hello")',
            explanation="Prints hello",
            usage_example='python script.py',
            dependencies=[],
        )
        assert artifact.title == "Hello World"
        assert artifact.language == "python"

    def test_graph_compiles_without_api_key(self) -> None:
        from vibe_agents.graphs.code_graph import build_code_graph

        graph = build_code_graph()
        assert graph is not None
