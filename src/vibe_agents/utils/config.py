"""
Configuration helpers.

Loads environment variables from .env (if present) and exposes typed
settings used across graphs and crews.
"""

import os

from dotenv import load_dotenv

# Load .env from the project root (one level above src/)
load_dotenv()


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key, raising a clear error if absent."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key or key == "your_anthropic_api_key_here":
        raise OSError(
            "ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add your key, "
            "or run with --mock to skip real API calls."
        )
    return key


def get_model_name() -> str:
    """Return the Claude model name, defaulting to claude-sonnet-4-5."""
    return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")


def is_mock_mode() -> bool:
    """Return True when MOCK_MODE env var is truthy."""
    return os.getenv("MOCK_MODE", "false").lower() in {"1", "true", "yes"}
