"""Tests for deterministic ingestion text normalization."""

from __future__ import annotations

import pytest

from ragplan.ingestion.normalize import normalize_text

pytestmark = pytest.mark.unit


def test_normalize_text_nfkc_line_endings_and_paragraphs() -> None:
    text = "  Ａ\tB\r\nwrapped  line\r\n\r\n\rsecond\u00a0 paragraph  \n\n\n"

    assert normalize_text(text) == "A B wrapped line\n\nsecond paragraph"


def test_normalize_text_removes_whitespace_only_input() -> None:
    assert normalize_text(" \t\r\n\u00a0 ") == ""


def test_normalize_text_requires_string() -> None:
    with pytest.raises(TypeError, match="text"):
        normalize_text(None)  # type: ignore[arg-type]
