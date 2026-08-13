"""Build the frozen Stage 2 benchmark from checksum-pinned upstream train data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ragplan.benchmark.builder import build_stage2
from ragplan.benchmark.download import prepare_raw_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download/resume checksum-pinned raw inputs before building",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    raw_dir = (args.raw_dir or repository_root / "benchmark/datasets/raw").resolve()
    if args.download:
        prepare_raw_datasets(raw_dir)
    result = build_stage2(
        repository_root=repository_root,
        raw_dir=raw_dir,
        model_snapshot=args.model_snapshot.resolve(),
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
