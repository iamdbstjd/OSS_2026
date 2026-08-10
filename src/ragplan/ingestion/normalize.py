"""Canonical text normalization used before deterministic chunking."""

from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    """Return normalized text while retaining meaningful paragraph boundaries.

    The pipeline applies Unicode NFKC, converts every line ending to ``\n``, and
    treats one or more blank lines as a paragraph separator.  Whitespace inside a
    paragraph (including ordinary line wraps) collapses to one ASCII space;
    paragraphs are then joined by exactly two newlines.  Thus leading/trailing
    whitespace and surplus blank lines cannot affect chunk IDs, while paragraphs
    remain distinct from ordinary wrapped lines.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    paragraphs: list[str] = []
    current_lines: list[str] = []
    for line in normalized.split("\n"):
        if line.strip():
            current_lines.append(line)
        elif current_lines:
            paragraphs.append(" ".join(" ".join(current_lines).split()))
            current_lines = []
    if current_lines:
        paragraphs.append(" ".join(" ".join(current_lines).split()))
    return "\n\n".join(paragraphs)
