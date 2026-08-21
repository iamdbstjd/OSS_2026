#!/usr/bin/env python3
"""Recover a missing local manifest from an already sealed Neo4j graph stage."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ragplan.backends.graph.neo4j import Neo4jGraphConfig, Neo4jGraphWriter
from ragplan.ingestion.manifest import write_contract_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-version", required=True)
    parser.add_argument("--graph-stage-manifest", type=Path, required=True)
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("RAGPLAN_GRAPH__URI", "bolt://127.0.0.1:7687"),
    )
    parser.add_argument("--neo4j-user", default=os.environ.get("RAGPLAN_GRAPH__USER", "neo4j"))
    parser.add_argument("--neo4j-database", default="neo4j")
    return parser


async def _execute(args: argparse.Namespace) -> object:
    password = os.environ.get("RAGPLAN_GRAPH__PASSWORD") or os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise ValueError("Neo4j password must be supplied through the environment")
    writer = Neo4jGraphWriter.connect(
        Neo4jGraphConfig(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=password,
            database=args.neo4j_database,
        )
    )
    try:
        manifest = await writer.recover_stage(args.corpus_version)
        write_contract_json(args.graph_stage_manifest, manifest)
        return manifest
    finally:
        await writer.close()


def main() -> None:
    manifest = asyncio.run(_execute(build_parser().parse_args()))
    print(json.dumps(manifest.model_dump(mode="json"), sort_keys=True))


if __name__ == "__main__":
    main()
