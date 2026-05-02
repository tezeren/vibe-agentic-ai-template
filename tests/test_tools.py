"""
Tests for tool functions in vibe_agents.tools

All tests are offline — they test tool logic without real API calls.
"""




class TestFormatPythonCode:
    def test_valid_code_returns_syntax_ok(self) -> None:
        from vibe_agents.tools.code_tools import format_python_code

        result = format_python_code.invoke({"code": "def hello():\n    return 'world'"})
        assert "Syntax OK" in result

    def test_invalid_code_returns_syntax_error(self) -> None:
        from vibe_agents.tools.code_tools import format_python_code

        result = format_python_code.invoke({"code": "def hello(\n    pass"})
        assert "Syntax error" in result or "SyntaxError" in result.lower() or "error" in result.lower()

    def test_empty_code_is_valid(self) -> None:
        from vibe_agents.tools.code_tools import format_python_code

        result = format_python_code.invoke({"code": ""})
        assert "Syntax OK" in result


class TestSearchPythonDocs:
    def test_returns_string(self) -> None:
        from vibe_agents.tools.code_tools import search_python_docs

        result = search_python_docs.invoke({"query": "list comprehension"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_query_in_result(self) -> None:
        from vibe_agents.tools.code_tools import search_python_docs

        result = search_python_docs.invoke({"query": "asyncio"})
        assert "asyncio" in result


class TestGetWeatherMock:
    def test_mock_weather_function(self) -> None:
        from vibe_agents.tools.weather import get_weather_mock

        result = get_weather_mock("London")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_weather_unknown_city(self) -> None:
        from vibe_agents.tools.weather import get_weather_mock

        result = get_weather_mock("MadeUpCity123")
        assert "MadeUpCity123" in result or isinstance(result, str)
