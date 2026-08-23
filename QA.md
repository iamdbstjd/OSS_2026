# RAGPlan Competition QA Plan

> Status: draft execution plan
> Scope: Stage 0-12 and the Stage 13 public surface
> Primary target: reproducible three-minute competition demonstration

## 1. Purpose

This plan verifies that the public claims used in the competition demonstration are backed by
repeatable behavior. Unit-test count alone is not a release decision. Every critical demo claim
must have a test case, an expected result, a recovery procedure, and retained evidence.

The QA boundary ends at ranked retrieval evidence. RAGPlan does not claim to generate an LLM
answer, provide authentication or multi-tenancy, or serve the Stage 12 `research_only` model.

## 2. Quality priorities

| Priority | Meaning | Release rule |
|---|---|---|
| P0 | A filmed claim or destructive recovery path | Every case must pass; no skips |
| P1 | Supporting contract, privacy, or reproducibility check | Must pass or have an accepted issue |
| P2 | Optional presentation polish or timing observation | May be deferred with a recorded reason |

## 3. Test environments

| Profile | Infrastructure | Purpose |
|---|---|---|
| `fast` | Python 3.12 only | Planner, evidence, privacy, and LLM-handoff contracts |
| `vector` | Qdrant plus the checksum-pinned MiniLM cache | Real sample ingest and semantic retrieval |
| `dual-store` | API, Qdrant, Neo4j, verified active corpus | Fixed Hybrid and public API behavior |
| `failure-injection` | `dual-store` plus Docker control permission | Neo4j outage, degraded service, and recovery |

Use WSL Ubuntu and the repository under `~/projects/OSS_2026`. Never record `.env`, credentials,
tokens, model cache paths containing personal information, or raw benchmark data.

## 4. Safety controls

1. Live tests are disabled unless `RAGPLAN_QA_LIVE=1` is explicitly set.
2. Neo4j is never stopped unless `RAGPLAN_QA_ALLOW_FAILURE_INJECTION=1` is also set.
3. The failure-injection test restarts Neo4j in a `finally` block and verifies recovery.
4. Tests never run `docker compose down --volumes`, delete a corpus, or modify frozen benchmark data.
5. Stage 12 remains `research_only`; public `cost_aware` serving must fail closed.
6. P0 live cases that are skipped do not count as a QA pass for filming.

## 5. QA matrix

| ID | Priority | Scenario | Expected result | Automation |
|---|---:|---|---|---|
| QA-001 | P0 | `demo-plan` at 50 ms | P0 vector, no retrieval, raw query absent | `tests/qa/test_demo_qa.py` |
| QA-002 | P0 | `demo-plan` at 500 ms | P1 vector, graph candidates fail closed | `tests/qa/test_demo_qa.py` |
| QA-003 | P1 | Stage 9/10 row claim | 224,640 + 199,680 = 424,320 | `tests/qa/test_demo_qa.py` |
| QA-004 | P0 | Stage 12 deployment claim | `research_only`, public serving disabled, guard tripped | `tests/qa/test_demo_qa.py` |
| QA-005 | P1 | Provider-neutral LLM handoff | Ranked evidence retained; no provider SDK | `tests/qa/test_demo_qa.py` |
| QA-101 | P0 | Public Compose surface | `/health`, `/ready`, `/metrics` satisfy v1 contracts | `tests/e2e/test_compose_search.py` |
| QA-102 | P0 | Fixed Hybrid P5 | Vector and graph succeed; `weighted_rrf_v1`; results non-empty | `tests/e2e/test_compose_search.py` |
| QA-103 | P1 | Metrics accounting | Search increments request/planner/latency counters | `tests/e2e/test_compose_search.py` |
| QA-104 | P0 | Public cost-aware request | HTTP 503 `MODE_UNAVAILABLE` | `tests/e2e/test_compose_search.py` |
| QA-105 | P0 | Vector quickstart | P0 result ranks the Ada Lovelace evidence first | `tests/e2e/test_compose_search.py` |
| QA-106 | P0 | Neo4j outage | `/ready=degraded`; Qdrant remains healthy; graph unavailable | `tests/e2e/test_compose_search.py` |
| QA-107 | P0 | Rule during Neo4j outage | Request succeeds through vector-only execution | `tests/e2e/test_compose_search.py` |
| QA-108 | P0 | Explicit graph during outage | Stable HTTP 503 error | `tests/e2e/test_compose_search.py` |
| QA-109 | P0 | Neo4j recovery | `/ready=ready`; graph modes return | `tests/e2e/test_compose_search.py` |
| QA-201 | P2 | Full filmed rehearsal | All scenes complete within 180 seconds | Manual stopwatch/video review |
| QA-202 | P0 | Secret and screen review | No password, token, `.env`, or personal path visible | Manual frame review |

## 6. Execution

Install the complete locked development environment:

```bash
uv sync --frozen --python 3.12 --all-groups
```

Run the fast QA suite without Docker or a model:

```bash
uv run pytest -q -m "qa and not e2e"
```

Run the standard quality gate:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not integration and not e2e"
docker compose config --quiet
```

Start a preconfigured dual-store demo runtime and run non-destructive live QA:

```bash
docker compose up -d --build --wait --wait-timeout 240
RAGPLAN_QA_LIVE=1 \
uv run pytest -q -m "qa and e2e" tests/e2e/test_compose_search.py
```

The canonical full demo run, including the model-backed quickstart and controlled failure injection,
is:

```bash
RAGPLAN_QA_LIVE=1 \
RAGPLAN_QA_RUN_QUICKSTART=1 \
RAGPLAN_QA_ALLOW_FAILURE_INJECTION=1 \
uv run pytest -q -m "qa and e2e" tests/e2e/test_compose_search.py
```

Before the full run, all Stage 6 runtime variables must point to the verified active corpus. The
default Compose configuration intentionally exposes a healthy but `not_ready` API when no runtime
profile is configured; that state is not sufficient for QA-102 or QA-106.

## 7. Demo-specific assertions

- The 50 ms and 500 ms planner examples choose P0 and P1 respectively, but neither demonstrates
  automatic graph routing while the human graph audit is incomplete.
- The Fixed Hybrid scene is an explicit `planner=fixed_hybrid`, `plan_id=P5` request.
- A Neo4j outage is healthy degradation only when Qdrant and the active corpus remain usable.
- The Stage 12 evidence is a rejection demonstration, not a successful model-serving claim.
- `424,320` is the sum of Stage 9 baseline and Stage 10 profiler trial rows, not one benchmark run.
- The LLM example builds messages but does not call an LLM provider.

## 8. Evidence retention

For every release candidate, record:

```text
commit SHA
UTC timestamp
OS / Python / uv / Docker versions
QA command and exit code
pytest summary
/ready before failure injection
/ready during outage
/ready after recovery
redacted Fixed Hybrid response
Stage 12 evidence manifest checksum
manual rehearsal duration
```

Generated logs belong under `artifacts/qa/<run-id>/` and are ignored by Git by default. Commit only a
small redacted summary after review; never force-add raw queries, secrets, model files, or benchmark
raw rows.

## 9. Release gate

The demonstration is approved only when:

- all P0 cases pass without skips;
- Neo4j is confirmed recovered after failure injection;
- formatting, lint, type checking, and non-integration tests pass;
- the displayed test count comes from the same commit being filmed;
- the working tree and frozen evidence identities are recorded;
- the manual rehearsal finishes within 180 seconds;
- a final frame-by-frame secret review passes.

If any condition fails, mark the run `BLOCKED`, retain the failure evidence, and do not replace the
result with an unmeasured claim.

## 10. Current local execution record

Date: 2026-08-23
Base commit: `9ee5a55`

| Check | Result | Evidence |
|---|---|---|
| New fast QA | PASS | 5 passed |
| Non-integration regression | PASS | 399 passed, 1 skipped, 18 deselected |
| QA file formatting and lint | PASS | 3 files formatted; Ruff checks passed |
| Live E2E safety guard | PASS | 5 tests skipped without opt-in; no service mutation |
| Type checking | BLOCKED | QA-BUG-001 |
| Live Compose QA | BLOCKED | QA-ENV-001 |

### QA-BUG-001 — Stage 11 `skops` mypy import classification

`uv run mypy src` reports three errors at `src/ragplan/planner/artifacts.py:16`. The installed
`skops` runtime works and its tests pass, but mypy reports `import-not-found` while the source only
ignores `import-untyped`. This pre-existing defect blocks the full release gate until fixed and
reviewed.

### QA-ENV-001 — Docker unavailable inside the active WSL distribution

The WSL shell currently reports that the `docker` command is unavailable and recommends enabling
Docker Desktop WSL integration. Live Qdrant, Hybrid, outage, and recovery QA were therefore not run.
No container was stopped or modified. Resolve the Docker integration prerequisite and verify the
Stage 6 active corpus before enabling live or failure-injection QA.
