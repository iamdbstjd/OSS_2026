"""Checksum-pinned, resumable downloads for the non-redistributed Stage 2 sources."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, Any, Final


class DatasetDownloadError(RuntimeError):
    """Raised when a raw dataset cannot be downloaded or verified safely."""


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int


RAW_DOWNLOADS: Final = (
    DownloadSpec(
        key="dpr_nq_train",
        filename="biencoder-nq-train.json.gz",
        url="https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz",
        sha256="3249231587e8140e3794c060b0233afc61f4fa5e40b6a172d59519af5fe40c73",
        size_bytes=2_314_892_908,
    ),
    DownloadSpec(
        key="hotpot_train_shard_0",
        filename="hotpot_train_00000.parquet",
        url=(
            "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
            "1908d6afbbead072334abe2965f91bd2709910ab/"
            "distractor/train-00000-of-00002.parquet?download=true"
        ),
        sha256="76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
        size_bytes=165_624_177,
    ),
    DownloadSpec(
        key="hotpot_train_shard_1",
        filename="hotpot_train_00001.parquet",
        url=(
            "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
            "1908d6afbbead072334abe2965f91bd2709910ab/"
            "distractor/train-00001-of-00002.parquet?download=true"
        ),
        sha256="713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f",
        size_bytes=166_162_479,
    ),
    DownloadSpec(
        key="musique_v1_0",
        filename="musique_v1.0.zip",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h&export=download&confirm=t"
        ),
        sha256="98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd",
        size_bytes=272_049_578,
    ),
)

MUSIQUE_MEMBERS: Final = (
    "data/musique_ans_v1.0_train.jsonl",
    "data/dev_test_singlehop_questions_v1.0.json",
)


def prepare_raw_datasets(cache_dir: Path) -> tuple[Path, ...]:
    """Download every pinned source and extract only required MuSiQue train metadata."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(download_verified(spec, cache_dir=cache_dir) for spec in RAW_DOWNLOADS)
    extract_musique(paths[-1], destination=cache_dir / "musique")
    return paths


def download_verified(spec: DownloadSpec, *, cache_dir: Path) -> Path:
    """Resume into ``.part``, verify size/SHA-256, then atomically publish the file."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / spec.filename
    if destination.is_file():
        verify_file(destination, expected_sha256=spec.sha256, expected_size=spec.size_bytes)
        return destination

    partial = destination.with_name(f"{destination.name}.part")
    if partial.is_file() and partial.stat().st_size == spec.size_bytes:
        verify_file(partial, expected_sha256=spec.sha256, expected_size=spec.size_bytes)
        os.replace(partial, destination)
        return destination
    if partial.is_file() and partial.stat().st_size > spec.size_bytes:
        raise DatasetDownloadError(f"partial dataset artifact is oversized: {spec.filename}")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        spec.url,
        headers={
            "User-Agent": "RAGPlan/0.1 Stage2 dataset downloader",
            **({"Range": f"bytes={start}-"} if start else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            resumed = start > 0 and response.status == 206
            mode = "ab" if resumed else "wb"
            with partial.open(mode) as output:
                _copy_stream(response, output)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise DatasetDownloadError(f"download failed for {spec.key}") from exc

    verify_file(partial, expected_sha256=spec.sha256, expected_size=spec.size_bytes)
    os.replace(partial, destination)
    return destination


def verify_file(path: Path, *, expected_sha256: str, expected_size: int | None = None) -> str:
    """Fail closed on a missing, wrong-size, or checksum-mismatched artifact."""

    if not path.is_file():
        raise DatasetDownloadError(f"dataset artifact is missing: {path.name}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise DatasetDownloadError(f"dataset artifact size mismatch: {path.name}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise DatasetDownloadError(f"dataset artifact checksum mismatch: {path.name}")
    return actual


def verify_raw_datasets(cache_dir: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for spec in RAW_DOWNLOADS:
        path = cache_dir / spec.filename
        verify_file(path, expected_sha256=spec.sha256, expected_size=spec.size_bytes)
        paths.append(path)
    return tuple(paths)


def extract_musique(archive: Path, *, destination: Path) -> tuple[Path, ...]:
    """Extract the two allowlisted members; reject path traversal and archive drift."""

    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as source:
            names = set(source.namelist())
            missing = set(MUSIQUE_MEMBERS) - names
            if missing:
                raise DatasetDownloadError("MuSiQue archive is missing required members")
            for member in MUSIQUE_MEMBERS:
                relative = PurePosixPath(member)
                if relative.is_absolute() or ".." in relative.parts:
                    raise DatasetDownloadError("MuSiQue archive contains an unsafe path")
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_file, target.open("wb") as output_file:
                    shutil.copyfileobj(input_file, output_file)
                extracted.append(target)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetDownloadError("MuSiQue archive could not be extracted") from exc
    return tuple(extracted)


def file_sha256(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_stream(source: IO[Any], destination: IO[Any]) -> None:
    shutil.copyfileobj(source, destination, length=1024 * 1024)


def specs_by_key(specs: Iterable[DownloadSpec] = RAW_DOWNLOADS) -> dict[str, DownloadSpec]:
    return {spec.key: spec for spec in specs}
