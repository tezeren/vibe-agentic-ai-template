"""
Pytest configuration.

Ensures the test suite runs without any real API keys by providing
a session-scoped fixture that clears credentials from the environment.
Individual tests that DO need a mock key can use the `mock_api_key` fixture.
"""

import pytest


@pytest.fixture(autouse=True)
def _clear_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Auto-use fixture: removes ANTHROPIC_API_KEY from the environment for
    every test, ensuring tests never accidentally make real API calls.

    Tests that need to simulate a valid key can set it explicitly via
    monkeypatch.setenv() inside the test body.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
