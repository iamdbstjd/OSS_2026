[한국어](README.md) | **English**

# RAGPlan

Latency-budget-aware execution planning for vector and graph retrieval.

RAGPlan is an experimental, local-first retrieval optimizer. It selects and
executes vector, graph, fixed-hybrid, rule-based, or cost-aware retrieval plans
and returns ranked evidence with an execution trace. Answer generation, agent
loops, authentication, multi-tenancy, and production deployment are outside
the MVP scope.

## Implemented foundation

Stages 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9 provide the reproducible bootstrap, shared runtime
contracts, frozen benchmark truth, vector/graph retrieval, deterministic fixed-hybrid
fusion, budget-aware parallel execution, and explainable rule planning:

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
- a frozen 600-query `adaptive_rag_bench_v1` manifest and 360/120/120 group split
- graded chunk qrels from the production 220/40 chunker and pure Recall/MRR/nDCG functions
- checksum/license/attribution audit and a separate 100-query synthetic graph fixture
- pinned spaCy NER and deterministic SVO/passive/copular/appositional relation extraction
- a Neo4j writer with parameterized Cypher, idempotent batches, and transaction timeouts
- an atomic active-corpus pointer switched only after observed Qdrant/Neo4j ID reconciliation
- a 100-train-sentence human-review queue and fail-closed rule graph-tier policy
- exact normalized seeds, 1–3-hop `RELATES_TO` traversal, and separate `MENTIONS` recovery
- hard caps of 5 seeds/50 paths per seed/500 entities/100 chunks and deadline-bound transactions
- direction-preserving path provenance, Graph Score V0 contributions, and graph phase traces
- `weighted_rrf_v1` with canonical chunk deduplication and per-source rank/score/contribution
- one vector/graph/fixed-hybrid engine supporting P4/P5/P6/P8 with P5 as the default
- a two-branch parallel scheduler with `runtime_semantics_version=v1` state evidence
- typed partial responses that preserve the successful branch ranking on sibling failure
- 32-request admission, per-backend five-failure/30-second circuits, and ingress kill switches
- traces separating backend-client timeout from application deadline with disconnect cleanup
- a single-pass query analyzer using the frozen `qf_v1` schema and versioned regex/keyword config
- a rule planner selecting P0/P1/P4/P5/P6/P8 from remaining budget, graph safety, and static p95 profiles
- a resumable, validated Stage 9 harness for the 224,640-row baseline matrix on one production engine

The learned cost-aware planner remains a later stage. A Stage 3 corpus is deliberately `vector_staged`;
only successful Stage 4 dual-store verification makes it `active`. The rule graph tier
stays disabled until the human extraction audit is complete.

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
  --stage-manifest artifacts/sample-stage3-vector.json \
  --chunks-output artifacts/sample-stage3-chunks.jsonl
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

## Stage 4 graph ingestion and activation

Process the exact Chunk JSONL written by Stage 3 through the pinned spaCy pipeline,
then stage an inactive graph version in Neo4j. Supply the password only through the
environment, never as a command argument.

```bash
docker compose up -d neo4j
uv sync --frozen --group graph-extraction

export RAGPLAN_GRAPH__PASSWORD=ragplan-demo-change-me

uv run python scripts/ingest_graph.py \
  --chunks artifacts/sample-stage3-chunks.jsonl \
  --vector-stage-manifest artifacts/sample-stage3-vector.json \
  --graph-stage-manifest artifacts/sample-stage4-graph.json

uv run python scripts/activate_corpus.py \
  --vector-stage-manifest artifacts/sample-stage3-vector.json \
  --graph-stage-manifest artifacts/sample-stage4-graph.json \
  --manifest-root artifacts/ingestion \
  --ingestion-run-id sample-stage4-run-v1 \
  --source-dataset ragplan-stage3-sample \
  --source-version v1 \
  --source-sha256 7d7b70f6ac6b6e9cd8053efec5526a797b57f482a678267f4faea66219cf54de
```

The graph writer is an idempotent no-op for identical content under the same corpus
version and rejects a changed graph checksum before writing. Activation re-reads both
backends, compares chunk counts and canonical-ID checksums, and only then atomically
replaces the local active pointer. A failure on either side preserves the old active
version; any previously verified run can be selected for rollback.
Re-running `ingest_graph.py` with identical input idempotently retries a failed partial
batch. Use `scripts/manage_ingestion.py` for explicit rollback, record removal, or
inactive-store discard.

To serve the activated corpus through the API vector path, leave every Stage 3
runtime variable empty and set all five Stage 4 values documented in `.env.example`.
Startup fails unless the active pointer and immutable run manifest match the vector
stage's corpus, count, checksum, and model revision exactly.

The hash-frozen 100-sentence audit queue from real Stage 2 train passages is under
`benchmark/audits/graph_extraction_v1/`. Human review and 20-sentence double review
are not complete, so `configs/graph_tier_policy.json` intentionally disables the rule
graph tier. Do not turn that state into a pass until reviewers complete
`reviews_v1.jsonl`.
Then run `uv run python scripts/evaluate_graph_audit.py` to recompute metrics and policy.

## Stage 5 bounded graph retrieval

Explicit `graph` comparison mode runs only against an atomically activated corpus. Query seeds use
the same pinned spaCy and normalization pipeline as ingestion. A storage lookup succeeds only when
both the UUID and exact normalized alias match. Traversal discovers `RELATES_TO` edges in either
direction while preserving each returned relation's stored direction. `MENTIONS` participates only
after traversal, when chunks from the active corpus are recovered.

The graph-only path is executable from the CLI. Supply the password only through the environment,
never as a command argument. `top-k <= 20` selects P2 depth 1; larger values select P3 depth 3. A
request above the preset candidate count is recorded as a request-floor override in the trace.

```bash
export RAGPLAN_GRAPH__PASSWORD=ragplan-demo-change-me

uv run python scripts/search_graph.py \
  --query "How is Apple related to Beats Electronics?" \
  --manifest-root artifacts/ingestion \
  --graph-stage-manifest artifacts/sample-stage4-graph.json \
  --extractor-lockfile uv.lock \
  --top-k 21 \
  --latency-budget-ms 500
```

For the API graph-only profile, set all six Stage 5 variables in `.env.example` and leave Stage 3/4
runtime values empty. Startup re-verifies the active manifest, graph-stage evidence, extractor
version, and the live Neo4j marker's corpus/count/checksum. Unsupported-language graph requests fail
safely and will use vector-only once the rule planner is available. Explicit comparison mode remains
available while the human audit is pending, but automatic rule-planner graph selection stays off.

Run the real Neo4j 1/2/3-hop, cycle, direction, and provenance check with:

```bash
RAGPLAN_TEST_NEO4J_URI=bolt://127.0.0.1:7687 \
RAGPLAN_TEST_NEO4J_USER=neo4j \
RAGPLAN_TEST_NEO4J_PASSWORD=ragplan-demo-change-me \
uv run pytest -q tests/integration/test_graph_retrieval.py
```

## Stage 6 fixed hybrid and fusion

One `BaselineSearchEngine` runs explicit `vector`, `graph`, and `fixed_hybrid`
against the same active corpus. Fixed hybrid accepts P4/P5/P6/P8 and defaults to P5.
Vector and graph candidates are deduplicated only by canonical chunk ID, then each
source contributes `weight / (60 + 1-based rank)`. Ties resolve by ascending canonical
ID. Every final hit retains each source's original rank/score, weight, RRF contribution,
source metadata, and graph paths. Conflicting text, document ID, or shared document
metadata under the same ID fails with a corpus-consistency error.

The explicit comparison CLI first verifies that the active pointer, both stage manifests,
and the live stores all agree on corpus/count/checksum/model and extractor provenance.

```bash
export RAGPLAN_GRAPH__PASSWORD=ragplan-demo-change-me

uv run python scripts/search_hybrid.py \
  --query "How is Apple related to Beats Electronics?" \
  --mode fixed_hybrid \
  --plan-id P5 \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --vector-stage-manifest artifacts/sample-stage3-vector.json \
  --graph-stage-manifest artifacts/sample-stage4-graph.json \
  --manifest-root artifacts/ingestion \
  --extractor-lockfile uv.lock \
  --top-k 10 \
  --latency-budget-ms 5000
```

For the API, set all ten Stage 6 variables in `.env.example` and leave the Stage 3/4/5
profiles empty. This explicit comparison mode remains available while the human graph
audit is pending, but automatic graph selection by the rule planner remains disabled.
The Stage 7 scheduler now runs active branches concurrently under the same absolute deadline.

Run the real two-store vertical test with:

```bash
RAGPLAN_TEST_QDRANT_URL=http://127.0.0.1:6333 \
RAGPLAN_TEST_NEO4J_URI=bolt://127.0.0.1:7687 \
RAGPLAN_TEST_NEO4J_USER=neo4j \
RAGPLAN_TEST_NEO4J_PASSWORD=ragplan-demo-change-me \
uv run pytest -q tests/integration/test_fixed_hybrid.py
```

## Stage 7 scheduler, deadline, and graceful degradation

The Stage 6 dual-store profile now routes every explicit mode through one scheduler. A request
records `received → analyzing → planning → executing → fusing → complete|partial`, while each
branch records start/end boundaries, remaining budget, cancellation reason, timeout origin, and
circuit state. Two successful branches use Weighted RRF. One successful branch returns HTTP 200
`partial` in its original ranking when its sibling fails. All-deadline failure is 504, all-backend
failure is 503, and a normal zero-hit success is HTTP 200 `complete` with an empty result set.

The scheduler admits at most 32 in-flight requests per process and immediately returns
`OVERLOADED` instead of queueing. Each backend circuit opens for 30 seconds after five consecutive
transport/client-timeout failures and permits one half-open probe. Application-deadline
cancellation does not count as a circuit failure, and there is no request-internal retry. Qdrant
and Neo4j client timeout ceilings are 30 seconds; shorter transaction/request cutoffs for the
25–5000ms request budget derive from the scheduler's same absolute deadline.

`RAGPLAN_FORCE_VECTOR_ONLY=true` bypasses graph and cost-aware paths at ingress, while
`RAGPLAN_DISABLE_COST_AWARE=true` prevents cost-model selection. Both are snapshotted once per
request and recorded in the trace. HTTP client disconnect or parent cancellation cancels and
awaits every backend child, leaving no orphan task.

## Stage 8 query analyzer and rule planner

A `planner=rule` request produces normalized query metadata, language support, token/entity counts,
seed entities, and `qf_v1` signals once from the frozen `query_features_v1.json`; it also computes
the query embedding exactly once. Raw query text and embeddings are never stored in the trace. The
rule planner uses the post-analysis remaining branch budget and static p95 profiles from
`rule_planner_v1.json`. Relationship queries prefer P4/P5/P6 as budget permits, multi-hop queries
prefer P6/P8, and low budgets or unsupported languages safely reduce to P0. Threshold provenance
is frozen to `validation`, so a configuration cannot identify the test split as its tuning source.

If the human-audit gate in `configs/graph_tier_policy.json` fails or the graph circuit is open,
graph-enabled candidates are explicitly marked infeasible and only P0/P1 can execute. The checked-in
audit is intentionally pending, so default rule requests are vector-only. Every decision records its
selected plan, matched rules, post-analysis remaining budget, per-candidate feasibility, fallback
reason, feature/config version, and selection explanation in the trace.

## Reproduce the Stage 2 benchmark

Raw datasets are not committed. The command below resumes downloads from pinned
URLs, checks byte sizes and SHA-256 values, then scans the complete upstream
train pools and selects exactly 600 records by ascending
`SHA256(source_dataset + ":" + source_query_id + ":20260809)`.

```bash
uv sync --frozen --group benchmark --group graph-extraction
uv run python scripts/prepare_model.py --cache-dir models/minilm

MODEL_SNAPSHOT="models/minilm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"

uv run python scripts/prepare_benchmark.py \
  --download \
  --model-snapshot "$MODEL_SNAPSHOT"

uv run pytest -q tests/benchmark
```

The primary manifest contains 200 NQ, 200 Hotpot bridge, 100 Hotpot comparison,
and 100 exactly-three-hop MuSiQue queries. Splits preserve source quotas and
document/entity groups, and test IDs are frozen in a separate immutable
manifest. The 100-query synthetic graph covers cycles, hubs, disconnected
entities, and 1/2/3-hop paths, but is excluded from primary aggregates. Raw
licenses and checksums live in `benchmark/manifests/licenses.yaml`; exact and
near-duplicate handling is recorded in
`benchmark/manifests/corpus_policy_v1.json`. The byte sizes and SHA-256 values
of 15 core Stage 2 data/contract artifacts are bound into one generation by the
last-written `benchmark/manifests/artifact_set_v1.json` readiness record.

## Stage 9 baseline benchmark

Stage 9 compares only the frozen 480 train/validation queries through the same active corpus and
production engine. The matrix covers vector, graph depths 1/2/3, fixed P4/P5/P6/P8, and rule at
50/100/200/500ms, with one separate cold sweep, two warmups, and ten measured repetitions. The 120
held-out test queries are not loaded before Stage 14. Timeouts, backend errors, partial fallbacks,
and zero-result outcomes remain in raw rows and in latency/rate denominators.

First record the dedicated CPU host and container limits. Then set every Stage 6 variable from
`.env.example` to the exact active `adaptive_rag_bench_v1-corpus-v1` stores and start the resumable
run.

```bash
uv run ragplan benchmark capture-environment \
  --output artifacts/benchmark_environment.json \
  --container-resource-limits "qdrant=2cpu,4GiB;neo4j=2cpu,4GiB" \
  --confirm-dedicated

docker compose --profile benchmark build benchmark

docker compose --profile benchmark run --rm benchmark run \
  --run-id baseline_20260813 \
  --environment-manifest /opt/ragplan/artifacts/benchmark_environment.json \
  --confirm-dedicated

docker compose --profile benchmark run --rm benchmark aggregate \
  --run-id baseline_20260813
```

The runner fails closed on configuration or hardware/DB-tuning drift, held-out test access,
corpus/count/ID-checksum mismatch, concurrent writers, and reuse of a run ID with different
evidence. Completion writes append-only JSONL, raw/aggregate CSV, aggregate JSON, a validation-only
BestFixed lock, environment/run/protocol manifests, and checksums under
`benchmark/results/<run_id>/`. Aggregation uses type-7 p50/p95/p99 and 10,000 query-cluster paired
bootstrap samples (seed 20260809), without outlier removal. See `docs/benchmark.md` for the full
contract.

The repository does not fabricate the actual 480-query measurements. Run the command only after
the frozen corpus is ingested and activated in both stores and a dedicated measurement environment
is ready; large raw results are ignored by Git by default.

## Stage 10 offline plan profiler

Stage 10 exhaustively executes P0-enabled P0/P1/P2/P3/P4/P5/P6/P8 for each of the 480
train/validation queries at 50/100/200/500ms. Every query-plan-budget retains one cold run, two
warmups, and ten measured trials; both the loader and runner reject the held-out test split. The
public API plan contract remains unchanged. An internal benchmark entry point reuses the production
analyzer, absolute deadline, Stage 7 scheduler, fallback, and fusion path.

Use the same dedicated environment manifest and Stage 6 active-corpus configuration as Stage 9.

```bash
docker compose --profile benchmark run --rm benchmark profile \
  --run-id plans_20260819 \
  --environment-manifest /opt/ragplan/artifacts/benchmark_environment.json \
  --confirm-dedicated

docker compose --profile benchmark run --rm benchmark profile-aggregate \
  --run-id plans_20260819
```

Completion writes append-only raw trials, flattened query/plan feature CSV and canonical JSONL
training matrices, per-budget Oracle labels/distributions, and environment/derived-artifact
checksums under `benchmark/results/profile_<run_id>/`. Oracle selects maximum Recall@10 among
complete plan rows whose measured p95 execution latency is within budget, breaking ties by lower
p95, lower graph depth, then lower plan ID. Timeouts, errors, and trace/hash mismatches are retained
with explicit exclusion reasons.

## License

RAGPlan is licensed under Apache License 2.0. Third-party software, models, and
datasets retain their own licenses; see `THIRD_PARTY_LICENSES.md`.
