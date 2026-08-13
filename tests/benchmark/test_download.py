from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from ragplan.benchmark.download import (
    DatasetDownloadError,
    DownloadSpec,
    download_verified,
    extract_musique,
    verify_file,
)


def test_verify_file_fails_closed_on_checksum_and_size(tmp_path: Path) -> None:
    artifact = tmp_path / "source.bin"
    artifact.write_bytes(b"trusted")
    digest = hashlib.sha256(b"trusted").hexdigest()
    assert verify_file(artifact, expected_sha256=digest, expected_size=7) == digest

    with pytest.raises(DatasetDownloadError, match="size mismatch"):
        verify_file(artifact, expected_sha256=digest, expected_size=8)
    with pytest.raises(DatasetDownloadError, match="checksum mismatch"):
        verify_file(artifact, expected_sha256="0" * 64, expected_size=7)


def test_musique_extraction_is_an_explicit_allowlist(tmp_path: Path) -> None:
    archive = tmp_path / "musique.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("data/musique_ans_v1.0_train.jsonl", "{}\n")
        output.writestr("data/dev_test_singlehop_questions_v1.0.json", "{}")
        output.writestr("untrusted/extra.txt", "must not be extracted")
    extracted = extract_musique(archive, destination=tmp_path / "out")
    assert len(extracted) == 2
    assert not (tmp_path / "out/untrusted/extra.txt").exists()


def test_download_spec_is_immutable() -> None:
    spec = DownloadSpec("key", "file", "https://example.test/file", "0" * 64, 1)
    with pytest.raises(AttributeError):
        spec.size_bytes = 2  # type: ignore[misc]


def test_complete_verified_partial_is_recovered_without_network(tmp_path: Path) -> None:
    payload = b"complete interrupted download"
    spec = DownloadSpec(
        "key",
        "source.bin",
        "https://network-must-not-be-used.invalid/source.bin",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
    )
    (tmp_path / "source.bin.part").write_bytes(payload)
    destination = download_verified(spec, cache_dir=tmp_path)
    assert destination.read_bytes() == payload
    assert not (tmp_path / "source.bin.part").exists()
