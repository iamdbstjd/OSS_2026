"""Tests for deterministic tokenizer-based ingestion chunking."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ragplan.core.ids import canonical_document_id
from ragplan.ingestion.chunker import (
    ChunkerConfig,
    HuggingFaceTokenizerAdapter,
    TokenEncoding,
    chunk_document,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class FakeEncoding:
    tokens: tuple[str, ...]

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    def decode(self, start: int, end: int) -> str:
        return " ".join(self.tokens[start:end])


class FakeTokenizer:
    def encode(self, text: str) -> TokenEncoding:
        return FakeEncoding(tuple(text.replace("\n", " ").split()))


class FakeUncasedFastTokenizer:
    is_fast = True

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return " ".join(("ada", "lovelace", "wrote")[index] for index in token_ids)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        words = text.split()
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            end = start + len(word)
            offsets.append((start, end))
            cursor = end
        return {"input_ids": list(range(len(words))), "offset_mapping": offsets}


def test_hugging_face_adapter_preserves_source_case_with_fast_tokenizer_offsets() -> None:
    tokenizer = HuggingFaceTokenizerAdapter.from_tokenizer(FakeUncasedFastTokenizer())

    encoding = tokenizer.encode("Ada Lovelace wrote")

    assert encoding.token_count == 3
    assert encoding.decode(0, 3) == "Ada Lovelace wrote"
    assert encoding.decode(1, 3) == "Lovelace wrote"


def test_chunk_document_uses_token_windows_overlap_and_canonical_ids() -> None:
    chunks = chunk_document(
        source_dataset="dataset",
        source_document_id="document-1",
        corpus_version="2026-08",
        text="one two three four five six seven",
        tokenizer=FakeTokenizer(),
        config=ChunkerConfig(window_size=4, overlap=1),
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
    ]
    assert [chunk.token_count for chunk in chunks] == [4, 4]
    assert [chunk.position for chunk in chunks] == [0, 1]
    document_id = canonical_document_id("dataset", "document-1")
    assert all(chunk.document_id == document_id for chunk in chunks)
    assert chunks[0].id != chunks[1].id


def test_chunk_document_removes_duplicate_windows_before_assigning_positions() -> None:
    chunks = chunk_document(
        source_dataset="dataset",
        source_document_id="document-1",
        corpus_version="v1",
        text="repeat repeat repeat final",
        tokenizer=FakeTokenizer(),
        config=ChunkerConfig(window_size=2, overlap=1),
    )

    assert [chunk.text for chunk in chunks] == ["repeat repeat", "repeat final"]
    assert [chunk.position for chunk in chunks] == [0, 2]


@pytest.mark.parametrize("token_count", [4, 8])
def test_chunk_document_does_not_emit_an_overlap_only_tail(token_count: int) -> None:
    chunks = chunk_document(
        source_dataset="dataset",
        source_document_id="document-1",
        corpus_version="v1",
        text=" ".join(f"token-{index}" for index in range(token_count)),
        tokenizer=FakeTokenizer(),
        config=ChunkerConfig(window_size=4, overlap=1),
    )

    assert all(chunk.token_count > 1 for chunk in chunks)
    assert len(chunks) == (1 if token_count == 4 else 3)


@pytest.mark.parametrize(
    ("token_count", "expected_chunks"),
    [(220, 1), (221, 2), (400, 2), (401, 3)],
)
def test_default_220_40_boundaries(token_count: int, expected_chunks: int) -> None:
    chunks = chunk_document(
        source_dataset="dataset",
        source_document_id="boundary",
        corpus_version="v1",
        text=" ".join(f"token-{index}" for index in range(token_count)),
        tokenizer=FakeTokenizer(),
    )

    assert len(chunks) == expected_chunks
    assert all(chunk.token_count <= 220 for chunk in chunks)


def test_content_change_changes_chunk_id() -> None:
    arguments = {
        "source_dataset": "dataset",
        "source_document_id": "document-1",
        "corpus_version": "v1",
        "tokenizer": FakeTokenizer(),
    }

    first = chunk_document(text="same start", **arguments)
    second = chunk_document(text="changed start", **arguments)

    assert first[0].id != second[0].id


@pytest.mark.parametrize("window_size, overlap", [(0, 0), (220, -1), (220, 220), (True, 0)])
def test_chunker_config_rejects_invalid_overlap(window_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkerConfig(window_size=window_size, overlap=overlap)


def test_chunk_document_skips_empty_text() -> None:
    assert (
        chunk_document(
            source_dataset="dataset",
            source_document_id="document-1",
            corpus_version="v1",
            text=" \n\t ",
            tokenizer=FakeTokenizer(),
        )
        == ()
    )
