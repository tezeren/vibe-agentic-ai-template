"""
Tests for vibe_agents.utils.config

All tests run without API keys — they exercise config logic only.
"""


import pytest


class TestGetModelName:
    def test_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
        from vibe_agents.utils.config import get_model_name

        assert get_model_name() == "claude-sonnet-4-5"

    def test_respects_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-3-5")
        # Re-import to pick up the change (config reads os.getenv at call time)
        from vibe_agents.utils.config import get_model_name

        assert get_model_name() == "claude-haiku-3-5"


class TestGetApiKey:
    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from importlib import reload

        import vibe_agents.utils.config as cfg

        reload(cfg)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            cfg.get_anthropic_api_key()

    def test_raises_for_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "your_anthropic_api_key_here")
        from importlib import reload

        import vibe_agents.utils.config as cfg

        reload(cfg)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            cfg.get_anthropic_api_key()

    def test_returns_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-1234")
        from importlib import reload

        import vibe_agents.utils.config as cfg

        reload(cfg)
        assert cfg.get_anthropic_api_key() == "sk-ant-test-1234"


class TestIsMockMode:
    @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("MOCK_MODE", value)
        from importlib import reload

        import vibe_agents.utils.config as cfg

        reload(cfg)
        assert cfg.is_mock_mode() is True

    @pytest.mark.parametrize("value", ["false", "0", "no", ""])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("MOCK_MODE", value)
        from importlib import reload

        import vibe_agents.utils.config as cfg

        reload(cfg)
        assert cfg.is_mock_mode() is False
