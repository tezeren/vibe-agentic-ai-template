"""
Weather tool for LangGraph agents.

The real implementation calls a public weather API (no key required).
In --mock mode (or tests) it returns static fixture data so the project
works without network access.

Customise this file to swap in a different data source — e.g. OpenWeatherMap,
Tomorrow.io, or your own backend.
"""

import httpx
from langchain_core.tools import tool

from vibe_agents.utils.mock_data import get_mock_weather


@tool
def get_weather(city: str) -> str:
    """
    Fetch the current weather for *city* and return a human-readable summary.

    Args:
        city: The name of the city to look up (e.g. "London", "Tokyo").

    Returns:
        A plain-text weather summary.
    """
    try:
        # Open-Meteo is free and key-less — great for demos.
        # Docs: https://open-meteo.com/en/docs
        geo_url = (
            f"https://geocoding-api.open-meteo.com/v1/search"
            f"?name={city}&count=1&language=en&format=json"
        )
        with httpx.Client(timeout=10) as client:
            geo_resp = client.get(geo_url)
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return f"Could not find location data for '{city}'."

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weathercode"
            f"&wind_speed_unit=kmh"
        )
        with httpx.Client(timeout=10) as client:
            w_resp = client.get(weather_url)
            w_resp.raise_for_status()
            w_data = w_resp.json()

        current = w_data.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")

        return (
            f"Weather in {city}: {temp}°C, "
            f"humidity {humidity}%, wind {wind} km/h."
        )
    except Exception as exc:  # noqa: BLE001
        # Fall back to mock data on any network error (CI-friendly)
        mock = get_mock_weather(city)
        return f"[fallback] {mock['summary']} (error: {exc})"


def get_weather_mock(city: str) -> str:
    """Return mock weather without making any network calls — for tests / --mock mode."""
    mock = get_mock_weather(city)
    return mock["summary"]
