"""
Code-generation helper tools for LangGraph agents.

These are simple @tool-decorated functions the agent can call.
In a production app you might add tools to run tests, lint code,
read files, or call external APIs.
"""

from langchain_core.tools import tool


@tool
def format_python_code(code: str) -> str:
    """
    Attempt to auto-format Python code using the built-in `ast` module
    to verify syntax, then return the (unchanged) code with a status note.

    Args:
        code: The Python source code to check.

    Returns:
        A message indicating whether the code parsed successfully.
    """
    import ast

    try:
        ast.parse(code)
        return f"Syntax OK. Code:\n```python\n{code}\n```"
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}\nCode:\n```python\n{code}\n```"


@tool
def search_python_docs(query: str) -> str:
    """
    Search Python built-in documentation for a topic (stub implementation).

    In production, replace this with a real docs search, vector store
    retrieval, or API call.

    Args:
        query: The topic or symbol to look up.

    Returns:
        A short description (stub).
    """
    # Stub — returns a placeholder to show the tool-calling loop works.
    return (
        f"Documentation stub for '{query}': "
        "See https://docs.python.org/3/ for authoritative information."
    )
