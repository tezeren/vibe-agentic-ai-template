"""
Tests for vibe_agents.utils.mock_data

No API calls — pure unit tests.
"""

from vibe_agents.utils.mock_data import (
    MOCK_CODE_RESPONSE,
    MOCK_CODE_REVIEW,
    MOCK_WEATHER,
    get_mock_weather,
)


class TestGetMockWeather:
    def test_known_city_london(self) -> None:
        result = get_mock_weather("London")
        assert result["city"] == "London"
        assert isinstance(result["temperature_celsius"], int)
        assert "summary" in result

    def test_case_insensitive(self) -> None:
        lower = get_mock_weather("london")
        upper = get_mock_weather("LONDON")
        assert lower["city"] == upper["city"]

    def test_unknown_city_fallback(self) -> None:
        result = get_mock_weather("Atlantis")
        assert result["city"] == "Atlantis"
        assert "summary" in result

    def test_all_known_cities_have_required_fields(self) -> None:
        required = {"city", "temperature_celsius", "condition", "humidity_percent", "wind_kmh", "summary"}
        for city, data in MOCK_WEATHER.items():
            missing = required - data.keys()
            assert not missing, f"City '{city}' missing fields: {missing}"


class TestMockConstants:
    def test_code_response_is_python(self) -> None:
        assert "def " in MOCK_CODE_RESPONSE or "class " in MOCK_CODE_RESPONSE or "solution" in MOCK_CODE_RESPONSE

    def test_code_review_is_string(self) -> None:
        assert isinstance(MOCK_CODE_REVIEW, str)
        assert len(MOCK_CODE_REVIEW) > 0
