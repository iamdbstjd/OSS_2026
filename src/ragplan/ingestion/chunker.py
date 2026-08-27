"""Tokenizer-injected, deterministic text chunking."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from ragplan.core.ids import canonical_chunk_id, canonical_document_id
from ragplan.core.models import Chunk, ChunkerVersion
from ragplan.ingestion.normalize import normalize_text

type TokenId = int
type TokenOffset = tuple[int, int]
type OffsetEncoder = Callable[[str], tuple[TokenOffset, ...]]


class TokenEncoding(Protocol):
    """A lossless token encoding that can decode any half-open token range."""

    @property
    def token_count(self) -> int:
        """Return the number of model input tokens, excluding special tokens."""

    def decode(self, start: int, end: int) -> str:
        """Reconstruct the text represented by tokens in ``[start, end)``."""


class Tokenizer(Protocol):
    """Tokenizer abstraction that avoids loading or downloading a model."""

    def encode(self, text: str) -> TokenEncoding:
        """Encode normalized text without adding model-specific special tokens."""


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Token-window configuration for :func:`chunk_document`."""

    window_size: int = 220
    overlap: int = 40

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_size, bool)
            or not isinstance(self.window_size, int)
            or self.window_size < 1
        ):
            raise ValueError("window_size must be a positive integer")
        if isinstance(self.overlap, bool) or not isinstance(self.overlap, int):
            raise ValueError("overlap must be an integer")
        if not 0 <= self.overlap < self.window_size:
            raise ValueError("overlap must satisfy 0 <= overlap < window_size")


class HuggingFaceTokenizerAdapter:
    """Adapt a pinned SentenceTransformer/Hugging Face tokenizer to ``Tokenizer``.

    The wrapped tokenizer is supplied by the caller.  This module never imports a
    model library and never causes a model download.
    """

    def __init__(
        self,
        encode: Callable[[str], Sequence[TokenId]],
        decode: Callable[[Sequence[TokenId]], str],
        encode_offsets: OffsetEncoder | None = None,
    ) -> None:
        self._encode = encode
        self._decode = decode
        self._encode_offsets = encode_offsets

    @classmethod
    def from_tokenizer(cls, tokenizer: object) -> HuggingFaceTokenizerAdapter:
        """Build an adapter for a tokenizer exposing Hugging Face ``encode/decode``."""
        encode_method = getattr(tokenizer, "encode", None)
        decode_method = getattr(tokenizer, "decode", None)
        if not callable(encode_method) or not callable(decode_method):
            raise TypeError("tokenizer must expose callable encode and decode methods")

        def encode(text: str) -> Sequence[TokenId]:
            result = encode_method(text, add_special_tokens=False)
            if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
                raise TypeError("tokenizer.encode must return a sequence of token IDs")
            return result

        def decode(token_ids: Sequence[TokenId]) -> str:
            result = decode_method(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(result, str):
                raise TypeError("tokenizer.decode must return text")
            return result

        encode_offsets: OffsetEncoder | None = None
        call_method = getattr(tokenizer, "__call__", None)
        if getattr(tokenizer, "is_fast", False) is True and callable(call_method):

            def encode_offsets(text: str) -> tuple[TokenOffset, ...]:
                result = call_method(
                    text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                return _validated_offsets(result, text_length=len(text))

        return cls(encode=encode, decode=decode, encode_offsets=encode_offsets)

    def encode(self, text: str) -> TokenEncoding:
        """Encode with the frozen V1 token-ID decode contract."""

        return _TokenIdsEncoding(token_ids=tuple(self._encode(text)), decode_ids=self._decode)

    def encode_for_chunker(
        self,
        text: str,
        *,
        chunker_version: ChunkerVersion,
    ) -> TokenEncoding:
        """Select one explicit reconstruction contract without mutating V1."""

        if chunker_version is ChunkerVersion.TOKEN_DECODE_V1:
            return self.encode(text)
        if chunker_version is not ChunkerVersion.SOURCE_OFFSETS_V2:
            raise ValueError("unsupported chunker version")
        if self._encode_offsets is None:
            raise ValueError("chunker v2 requires a fast tokenizer with source offsets")
        return _SourceOffsetEncoding(
            source_text=text,
            offsets=self._encode_offsets(text),
        )


@dataclass(frozen=True, slots=True)
class _SourceOffsetEncoding:
    source_text: str
    offsets: tuple[TokenOffset, ...]

    @property
    def token_count(self) -> int:
        return len(self.offsets)

    def decode(self, start: int, end: int) -> str:
        if not 0 <= start <= end <= self.token_count:
            raise ValueError("token range is outside the encoding")
        if start == end:
            return ""
        return self.source_text[self.offsets[start][0] : self.offsets[end - 1][1]]


@dataclass(frozen=True, slots=True)
class _TokenIdsEncoding:
    token_ids: tuple[TokenId, ...]
    decode_ids: Callable[[Sequence[TokenId]], str]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def decode(self, start: int, end: int) -> str:
        return self.decode_ids(self.token_ids[start:end])


def _validated_offsets(value: object, *, text_length: int) -> tuple[TokenOffset, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("fast tokenizer call must return a mapping")
    token_ids = value.get("input_ids")
    raw_offsets = value.get("offset_mapping")
    if (
        not isinstance(token_ids, Sequence)
        or isinstance(token_ids, (str, bytes))
        or not isinstance(raw_offsets, Sequence)
        or isinstance(raw_offsets, (str, bytes))
    ):
        raise TypeError("fast tokenizer must return input_ids and offset_mapping sequences")
    if len(token_ids) != len(raw_offsets):
        raise ValueError("tokenizer input IDs and offsets must have equal lengths")

    offsets: list[TokenOffset] = []
    previous_end = 0
    for raw_offset in raw_offsets:
        if (
            not isinstance(raw_offset, Sequence)
            or isinstance(raw_offset, (str, bytes))
            or len(raw_offset) != 2
        ):
            raise TypeError("each tokenizer offset must contain start and end integers")
        start, end = raw_offset
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise TypeError("each tokenizer offset must contain start and end integers")
        if start < previous_end or not 0 <= start < end <= text_length:
            raise ValueError("tokenizer offsets must be ordered non-empty source ranges")
        offsets.append((start, end))
        previous_end = end
    return tuple(offsets)


def chunk_document(
    *,
    source_dataset: str,
    source_document_id: str,
    corpus_version: str,
    text: str,
    tokenizer: Tokenizer,
    config: ChunkerConfig = ChunkerConfig(),
    chunker_version: ChunkerVersion = ChunkerVersion.TOKEN_DECODE_V1,
) -> tuple[Chunk, ...]:
    """Normalize and divide a document into de-duplicated overlapping token windows.

    ``position`` is the original zero-based token-window ordinal. Duplicate
    decoded windows are omitted without renumbering later windows, so removing a
    duplicate cannot silently change the canonical IDs that follow it.
    """
    if not isinstance(corpus_version, str) or not corpus_version.strip():
        raise ValueError("corpus_version must be a non-empty string")
    if not isinstance(config, ChunkerConfig):
        raise TypeError("config must be a ChunkerConfig")

    normalized = normalize_text(text)
    if not normalized:
        return ()

    encoding = _encode_for_chunker(
        tokenizer,
        normalized,
        chunker_version=chunker_version,
    )
    if encoding.token_count < 0:
        raise ValueError("tokenizer returned a negative token count")

    document_id = canonical_document_id(source_dataset, source_document_id)
    chunks: list[Chunk] = []
    seen_text: set[str] = set()
    step = config.window_size - config.overlap
    for window_position, start in enumerate(range(0, encoding.token_count, step)):
        end = min(start + config.window_size, encoding.token_count)
        chunk_text = normalize_text(encoding.decode(start, end))
        if chunk_text and chunk_text not in seen_text:
            seen_text.add(chunk_text)
            chunks.append(
                Chunk(
                    id=canonical_chunk_id(document_id, window_position, chunk_text),
                    document_id=document_id,
                    corpus_version=corpus_version,
                    position=window_position,
                    text=chunk_text,
                    token_count=end - start,
                )
            )
        # Once a window reaches the end, another start would contain only the
        # overlap already represented by this final chunk.
        if end == encoding.token_count:
            break
    return tuple(chunks)


def _encode_for_chunker(
    tokenizer: Tokenizer,
    text: str,
    *,
    chunker_version: ChunkerVersion,
) -> TokenEncoding:
    if not isinstance(chunker_version, ChunkerVersion):
        raise TypeError("chunker_version must be a ChunkerVersion")
    if chunker_version is ChunkerVersion.TOKEN_DECODE_V1:
        return tokenizer.encode(text)
    versioned_encode = getattr(tokenizer, "encode_for_chunker", None)
    if not callable(versioned_encode):
        raise ValueError("chunker v2 requires a tokenizer with source-offset support")
    encoding = versioned_encode(text, chunker_version=chunker_version)
    if not hasattr(encoding, "token_count") or not callable(getattr(encoding, "decode", None)):
        raise TypeError("versioned tokenizer returned an invalid encoding")
    return cast(TokenEncoding, encoding)
