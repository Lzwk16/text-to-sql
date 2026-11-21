"""Prompt templates, stored as Markdown files next to this module.

Edit the ``.md`` files to change what the LLM is told. Placeholders use
``str.format`` syntax such as ``{question}``; a literal brace must be doubled.
"""

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the text of ``<name>.md`` (cached)."""
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **values: object) -> str:
    return load(name).format(**values)
