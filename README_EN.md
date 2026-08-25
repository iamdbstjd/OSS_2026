[한국어](README.md) | **English**

# RAGPlan

RAGPlan is a retrieval execution layer that selects vector, graph, or hybrid plans under a latency
budget and returns ranked evidence with an explainable trace.

It is not an end-user chatbot. RAGPlan sits in front of an existing LLM/RAG application and owns
retrieval strategy, deadlines, parallel branches, and fallback. Stages 0–12 and the Stage 13 public
surface are implemented. The learned cost-aware policy remains `research_only` after validation
gate failures; the default online planner remains Rule.

## Who it is for

- RAG developers changing vector/graph strategy inside a latency SLO
- AI infrastructure teams that need timeout, partial-result, and circuit-breaker semantics
- researchers reproducing a query×plan matrix and Oracle@Budget

The following are deliberately outside the current scope:

- a complete question-answering UI or LLM answer generation
- agent loops, authentication, multi-tenancy, or a distributed production control plane
- claims that Rule automatically routes to graph without the human graph audit
- public serving of the `research_only` cost model

## One-minute planner demo after install — no DB or model

```bash
uv sync --frozen
uv run ragplan demo-plan \
  --query "Who founded Acme and who acquired it?" \
  --budget-ms 100 \
  --entity-count 2 \
  --pretty

uv run ragplan verify --pretty
```

`demo-plan` never invents embeddings or retrieval results. It uses a lightweight token count plus
the caller-supplied entity count to explain `qf_v1` features and the Rule decision, and records
`mode=planner_only_no_embedding` and `executes_retrieval=false`.

## Infrastructure levels

| Goal | Required components |
|---|---|
| Validate code or inspect planning | Python 3.12 + `uv`; no DB/model |
| Vector sample ingest/search | Qdrant + MiniLM (checksum-pinned) |
| Full vector/graph/hybrid runtime | Qdrant + Neo4j + active corpus |
| Build a new graph corpus | The above + pinned spaCy `en_core_web_sm` |
| Production auth, multi-tenancy, agents | Not supported |

spaCy is required while extracting a new graph corpus, not while searching an existing vector-only
corpus.

## One sample ingest → search path

The password in `.env.example` is only for an isolated local demo.

```bash
cp .env.example .env
docker compose up -d qdrant
uv run ragplan quickstart-vector --pretty
```

This downloads only the approved MiniLM revision, verifies every SHA-256, idempotently ingests the
three-document sample into Qdrant, and executes one search through the production
`VectorSearchEngine`. Use `ragplan ingest`, `ragplan search`, and
`ragplan verify --configured-runtime` when running the steps separately.

## REST status and metrics

```bash
curl --fail http://127.0.0.1:8000/health
curl --silent --show-error http://127.0.0.1:8000/ready
curl --silent --show-error http://127.0.0.1:8000/metrics
```

- `/health` checks process liveness only.
- `/ready` reports corpus/backend capability and returns vector-only `degraded` when Neo4j fails.
- `/metrics` exposes request/result/error/planner/latency JSON metrics without raw query/entity labels.

## Hand results to an LLM

RAGPlan does not own answer generation. The [generic handoff example](examples/llm_handoff.py)
converts ranked `SearchResponse` chunks into provider-neutral messages without importing any LLM
SDK.

```bash
uv run ragplan search \
  --query "What did Ada Lovelace write about?" \
  --planner vector \
  --pretty > /tmp/ragplan-response.json

uv run python examples/llm_handoff.py \
  --response /tmp/ragplan-response.json \
  --question "What did Ada Lovelace write about?"
```

## Detailed implementation status

Stages 0–12 and the Stage 13 accessibility surface provide the reproducible bootstrap, shared
runtime contracts, frozen benchmark truth, vector/graph retrieval, deterministic hybrid fusion,
scheduling, Rule planning, research-only cost comparison, and public CLI/API surfaces:

- Python 3.12 package and `ragplan` CLI
- FastAPI search, liveness, readiness, and privacy-safe JSON metrics endpoints
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

The learned cost-aware planner reached offline comparison in Stages 11–12 but failed its model
gates and runtime guard, so public API activation remains disabled. A Stage 3 corpus is deliberately
`vector_staged`; only successful Stage 4 dual-store verification makes it `active`. The Rule graph
tier stays disabled until the human extraction audit is complete.

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

uv run ragplan ingest \
  --input examples/sample_corpus.json \
  --corpus-version sample-stage3-v2 \
  --model-cache models/minilm \
  --stage-manifest artifacts/sample-stage3-vector.json \
  --chunks-output artifacts/sample-stage3-chunks.jsonl

MODEL_SNAPSHOT="models/minilm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"
```

The ingest command downloads only the manifest allowlist for revision
`b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`, verifies every SHA-256, embeds all
document chunks in one batch, and verifies Qdrant's
schema, count, and canonical-ID checksum before writing the stage manifest atomically.
Repeating the same corpus version is an idempotent no-op; changing its canonical
ID set, embedding artifact, or embedding bytes is rejected before mutation.
The `vector_staged` v2 manifest binds the corpus to both the exact model
artifact SHA-256 and an aggregate checksum of every canonical-ID/vector pair.

Run a query through the same `VectorSearchEngine` path used by the injectable
FastAPI search endpoint:

```bash
RAGPLAN_STAGE3_MODEL_SNAPSHOT="$MODEL_SNAPSHOT" \
RAGPLAN_STAGE3_VECTOR_STAGE_MANIFEST=artifacts/sample-stage3-vector.json \
RAGPLAN_STAGE3_QDRANT_URL=http://127.0.0.1:6333 \
RAGPLAN_STAGE3_QDRANT_COLLECTION_PREFIX=ragplan_chunks \
uv run ragplan search \
  --query "Who wrote notes containing an algorithm for the Analytical Engine?" \
  --planner vector \
  --top-k 3 \
  --budget-ms 5000
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
  --ingestion-run-id sample-stage4-run-v2 \
  --source-dataset ragplan-stage3-sample \
  --source-version v2 \
  --source-sha256 55689e27fe1bae8f409182b312263a25cf0f8904613d146b3bed4c2f7945013f
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
  --query "Who collaborated with Ada Lovelace?" \
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
  --query "Who collaborated with Ada Lovelace?" \
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

## Stage 11 cost-model training

Stage 11 joins Stage 10 query/plan features into a deterministic numeric/one-hot schema and trains
Recall@10 quality plus conditional-p95 execution-latency models without raw embeddings. Query-level
train/validation separation is preserved and the held-out test split is never loaded.

```bash
uv run python scripts/train_cost_models.py \
  --matrix benchmark/results/profile_plans_audit_bypass_20260820_r2/training_matrix.jsonl \
  --profile-run-manifest benchmark/results/profile_plans_audit_bypass_20260820_r2/run_manifest.json \
  --profile-environment benchmark/results/profile_plans_audit_bypass_20260820_r2/environment.json \
  --oracle benchmark/results/profile_plans_audit_bypass_20260820_r2/oracle_at_budget.json \
  --stage9-raw benchmark/results/baseline_audit_bypass_20260820_r2/raw_trials.jsonl \
  --output-dir artifacts/cost_models/stage11_r2
```

Only checksum-first `.skops` artifacts are accepted. Types outside the repository allowlist,
critical runtime mismatches, and corrupted checksums are rejected. Current R2 validation reached
0.0092 quality MAE and 0.9284 latency coverage, but failed the 0.70 plan-pair ranking gate at
0.6815, per-plan latency coverage because P0/P1 have no latency labels, and the 0.10 pinball
improvement gate at 0.0479. The artifacts are therefore `research_only` and explicitly blocked from
serving. See `docs/model_training.md` and
`benchmark/manifests/stage11_model_evidence_r2.json`.

## Stage 12 research-only/offline policy comparison

Stage 12 scores all eight P0 plans only against validation evidence and does not connect the Stage
11 R2 artifacts to public search. Because the Stage 11 schema requires dataset source and query
tags that normal requests do not carry, `OfflineResearchContext` makes that dependency explicit;
public `planner=cost_aware` continues to return `MODE_UNAVAILABLE`.

```bash
uv run python scripts/evaluate_cost_policy.py
```

The actual R2 comparison evaluated 3,840 candidates across 480 query-budget groups. The research
policy produced Recall@10 +0.005507 versus Rule, +0.000726 versus BestFixed, and 0.002618 Oracle
regret. However, 1,856 candidate predictions were outside their valid range and 120 groups had no
predicted-feasible plan. The rolling calibration guard also disabled the artifact after the 319th
evaluable observation when p95 underprediction reached 0.21, routing the remaining 422 groups to
Rule. The model and result therefore remain `research_only`, not evidence for serving activation.
See `docs/offline_cost_policy.md` and
`benchmark/manifests/stage12_policy_evidence_r2.json` for the full contract and checksums.

## License

RAGPlan is licensed under Apache License 2.0. Third-party software, models, and
datasets retain their own licenses; see `THIRD_PARTY_LICENSES.md`.
