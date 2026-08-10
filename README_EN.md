[한국어](README.md) | **English**

# RAGPlan

Latency-budget-aware execution planning for vector and graph retrieval.

RAGPlan is an experimental, local-first retrieval optimizer. It selects and
executes vector, graph, fixed-hybrid, rule-based, or cost-aware retrieval plans
and returns ranked evidence with an execution trace. Answer generation, agent
loops, authentication, multi-tenancy, and production deployment are outside
the MVP scope.

## Implemented foundation

Stages 0, 1, and 3 provide the reproducible bootstrap, shared runtime
contracts, and a working vector-retrieval vertical slice:

- Python 3.12 package and `ragplan` CLI
- FastAPI liveness endpoint
- pinned dependency lockfile
- local Qdrant and Neo4j services through Docker Compose
- lint, type-check, and test configuration
- open-source contribution, security, and license files
- frozen Pydantic domain/request/response/trace contracts
- deterministic canonical document, chunk, entity, and Qdrant IDs
- the validated P0–P8 plan catalog with canonical SHA-256 identity
- one injectable monotonic absolute deadline with an ADR-010 reserve
- stable error/HTTP mappings and separate runtime/ingestion backend protocols
- deterministic Unicode normalization and 220-token chunks with 40-token overlap
- checksum-verified, local-only `all-MiniLM-L6-v2` embeddings at the pinned revision
- corpus-version-isolated Qdrant collections with UUIDv5 points and exact reconciliation
- explicit vector-mode execution with analysis, embedding, Qdrant, and total latency traces
- reproducible model preparation, sample ingestion, and search commands

Graph ingestion, graph retrieval, hybrid fusion, and adaptive planners remain
later stages. A Stage 3 corpus is deliberately marked `vector_staged`, not
`active`; activation requires the Stage 4 dual-store reconciliation contract.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker Engine with Docker Compose v2

## Local setup

```bash
cp .env.example .env
uv sync --frozen
uv run ragplan --version
```

The values in `.env.example` are for an isolated local demo only. Change the
Neo4j password before using any shared environment, and never commit `.env`.

## Start the local stack

```bash
docker compose config --quiet
docker compose up -d --build
curl --fail http://127.0.0.1:8000/health
```

The API binds to `127.0.0.1` by default. Qdrant and Neo4j are also exposed only
on loopback by the supplied Compose configuration.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not integration and not e2e"
```

The real Qdrant vector integration can be run while the local Qdrant service is
healthy:

```bash
RAGPLAN_TEST_QDRANT_URL=http://127.0.0.1:6333 \
RAGPLAN_TEST_MODEL_SNAPSHOT="$PWD/models/minilm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0" \
uv run pytest \
  tests/integration/test_qdrant_vector_backend.py \
  tests/integration/test_vector_vertical_slice.py
```

## Configuration

Non-secret defaults live in `configs/default.yaml`. Environment variables
override nested keys with the `RAGPLAN_` prefix and a double underscore, for
example:

```text
server.port       -> RAGPLAN_SERVER__PORT
vector.url        -> RAGPLAN_VECTOR__URL
graph.password    -> RAGPLAN_GRAPH__PASSWORD
```

Passwords and other secrets must be supplied through the environment or an
external secret manager; they do not belong in committed YAML files.

## Stage 1 contracts

Static retrieval plans are defined in `configs/plans.yaml`. P7 is retained as a
P1 reranker plan but excluded from the default P0 plan space. The loader rejects
unknown or missing fields, duplicate IDs, and branch/weight/depth/rerank
invariant violations before the API application is created.

Requests use the explicit planner modes `vector`, `graph`, `fixed_hybrid`,
`rule`, and `cost_aware`; `adaptive` is not an alias. Query text is trimmed and
limited to 1–4096 Unicode code points, request bodies to 32 KiB, top-k to 1–50,
and latency budgets to 25–5000 ms. P0 cost-aware requests require top-k 10.

All online stages receive the same monotonic `Deadline`. Branch work stops at
the absolute deadline minus `min(20 ms, max(5 ms, budget * 0.05))`; there is no
hidden grace interval. Default traces store a query SHA-256 and feature summary,
never the raw query or query embedding.

## Stage 3 vector demo

Start Qdrant, provision the exact allowlisted model snapshot, and ingest the
included three-document corpus:

```bash
docker compose up -d qdrant

uv run python scripts/prepare_model.py --cache-dir models/minilm

MODEL_SNAPSHOT="models/minilm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"

uv run python scripts/ingest.py \
  --input examples/sample_corpus.json \
  --corpus-version sample-stage3-v1 \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --stage-manifest artifacts/sample-stage3-vector.json
```

The model command downloads only the manifest allowlist for revision
`b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`, then verifies every file SHA-256.
The ingest command embeds all document chunks in one batch, verifies Qdrant's
schema, count, canonical-ID checksum, and writes the stage manifest atomically.
Repeating the same corpus version is an idempotent no-op; changing its canonical
ID set, embedding artifact, or embedding bytes is rejected before mutation.
The `vector_staged` v2 manifest binds the corpus to both the exact model
artifact SHA-256 and an aggregate checksum of every canonical-ID/vector pair.

Run a query through the same `VectorSearchEngine` path used by the injectable
FastAPI search endpoint:

```bash
uv run python scripts/search.py \
  --query "Who wrote notes containing an algorithm for the Analytical Engine?" \
  --stage-manifest artifacts/sample-stage3-vector.json \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --top-k 3 \
  --latency-budget-ms 5000
```

The response is JSON with ranked chunks and a redacted trace. It contains the
query hash and analysis/embedding/vector/total timings, but not the raw query or
query embedding.

To expose that verified stage through the Docker-served API, set all four
optional Stage 3 values in `.env` to their container paths, then restart the API:

```text
RAGPLAN_STAGE3_MODEL_SNAPSHOT=/opt/ragplan/models/minilm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0
RAGPLAN_STAGE3_VECTOR_STAGE_MANIFEST=/opt/ragplan/artifacts/sample-stage3-vector.json
RAGPLAN_STAGE3_QDRANT_URL=http://qdrant:6333
RAGPLAN_STAGE3_QDRANT_COLLECTION_PREFIX=ragplan_chunks
```

```bash
docker compose up -d --build api

curl --fail --show-error http://127.0.0.1:8000/v1/search \
  --header 'content-type: application/json' \
  --data '{"query":"Who wrote the first computer algorithm?","planner":"vector","top_k":3,"latency_budget_ms":5000}'
```

The default application remains `NOT_READY` when all four values are absent. A
partial or invalid explicit configuration fails startup, and a configured
startup verifies the local model checksums plus Qdrant schema, counts, IDs, and
embedding provenance before serving. It never promotes vector-only staging
evidence to the dual-store active pointer.

## License

RAGPlan is licensed under Apache License 2.0. Third-party software, models, and
datasets retain their own licenses; see `THIRD_PARTY_LICENSES.md`.
