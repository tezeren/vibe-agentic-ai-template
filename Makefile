.PHONY: install install-dev lint format test run-weather run-code run-crew clean help

# ── Default target ──
help:
	@echo "vibe-agentic-ai — available commands:"
	@echo ""
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install dev + production dependencies"
	@echo "  make lint          Run ruff linter"
	@echo "  make format        Run ruff formatter"
	@echo "  make test          Run pytest (no API calls required)"
	@echo "  make run-weather   Run LangGraph weather demo (mock mode)"
	@echo "  make run-code      Run LangGraph code-gen demo (mock mode)"
	@echo "  make run-crew      Run CrewAI code-gen demo (mock mode)"
	@echo "  make clean         Remove build artefacts"

# ── Install ──
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ── Lint / format ──
lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# ── Tests ──
test:
	pytest tests/ -v

# ── Run examples (mock mode — no API key needed) ──
run-weather:
	python -m vibe_agents.cli langgraph-weather --city "London" --mock

run-code:
	python -m vibe_agents.cli langgraph-code --prompt "Write a Python function to reverse a string" --mock

run-crew:
	python -m vibe_agents.cli crewai-code --prompt "Write a Python function to check if a number is prime" --mock

# ── Clean ──
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null; true
	rm -rf dist/ build/
