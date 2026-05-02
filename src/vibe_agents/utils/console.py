"""
Rich-powered console helpers for pretty CLI output.
"""

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()


def print_header(title: str, subtitle: str = "") -> None:
    """Print a styled header panel."""
    content = Text(title, style="bold cyan")
    if subtitle:
        content.append(f"\n{subtitle}", style="dim")
    console.print(Panel(content, border_style="cyan"))


def print_result(label: str, content: str, language: str = "text") -> None:
    """Print a labelled result, with syntax highlighting for code."""
    console.print(f"\n[bold green]{label}[/bold green]")
    if language != "text":
        syntax = Syntax(content, language, theme="monokai", line_numbers=False)
        console.print(Panel(syntax, border_style="green"))
    else:
        console.print(Panel(content, border_style="green"))


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_info(message: str) -> None:
    """Print an informational message."""
    console.print(f"[dim]{message}[/dim]")
