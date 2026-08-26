<p align="center">
  <strong>한국어</strong> · <a href="README_EN.md">English</a>
</p>

<h1 align="center">RAGPlan</h1>

<p align="center">
  <strong>질문마다 알맞은 검색 계획을, 주어진 latency budget 안에서.</strong><br>
  Vector · Graph · Hybrid retrieval을 선택하고 실행하는 오픈소스 retrieval control plane
</p>

<p align="center">
  <a href="https://github.com/iamdbstjd/OSS_2026/actions/workflows/ci.yml"><img src="https://github.com/iamdbstjd/OSS_2026/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2563EB.svg" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/version-v0.1.0-0EA5E9.svg" alt="v0.1.0">
  <img src="https://img.shields.io/badge/cost--aware-research__only-F59E0B.svg" alt="Cost-aware research only">
</p>

<p align="center">
  <a href="#빠른-시작">빠른 시작</a> ·
  <a href="#작동-방식">작동 방식</a> ·
  <a href="#검색-모드">검색 모드</a> ·
  <a href="#cli">CLI</a> ·
  <a href="#검증-결과">검증 결과</a> ·
  <a href="#기여하기">기여하기</a>
</p>

<p align="center">
  <img src="image/4.png" width="100%" alt="RAGPlan이 질문과 200ms 예산에 맞춰 vector, graph 또는 hybrid 검색을 선택하고 근거를 반환하는 모습">
</p>

<p align="center">
  <em>RAGPlan은 또 하나의 RAG 챗봇이 아닙니다. LLM이 어떤 근거를 받을지 결정하고, 실행하고, 검증합니다.</em>
</p>

---

## RAGPlan이 하는 일

RAGPlan은 질문과 latency budget을 입력받아 검색 계획을 선택하고, Qdrant와 Neo4j를
deadline 안에서 실행한 뒤 순위화된 근거와 설명 가능한 trace를 반환합니다.

답변 생성은 특정 LLM에 종속시키지 않습니다. OpenAI, 로컬 모델 또는 기존 RAG 파이프라인
앞에 배치하여 **검색 전략·deadline·병렬 branch·fallback**을 담당합니다.

```text
Question + latency budget
          │
          ▼
   Query Analyzer ──► Rule Planner
                          │
                          ▼
                Deadline-aware Scheduler
                    ┌─────┴─────┐
                    ▼           ▼
            MiniLM + Qdrant   Neo4j Graph
                    └─────┬─────┘
                          ▼
                    Weighted RRF
                          ▼
              Ranked Evidence + Redacted Trace
                          ▼
                       Any LLM
```

## What You Get

| 기능 | 제공하는 것 |
|---|---|
| Budget-aware planning | 남은 시간, 질의 특성, backend 상태에 따라 실행 가능한 plan 선택 |
| Vector · Graph · Hybrid | MiniLM 의미 검색, bounded graph traversal, 병렬 검색과 Weighted RRF |
| Deadline semantics | 요청 ingress부터 응답 완성까지 하나의 monotonic absolute deadline 적용 |
| Graceful degradation | 한 backend가 실패해도 성공한 branch의 검색 결과를 보존 |
| Fail-closed model serving | 검증 gate를 통과하지 못한 학습 모델을 public runtime에서 차단 |
| Evidence-first observability | 선택 이유, branch별 latency, fallback과 provenance를 redacted trace로 기록 |
| Reproducible evaluation | 고정 query·plan·budget matrix, Oracle@Budget, checksum 기반 evidence |

## 빠른 시작

### 1분 planner-only 데모

DB, embedding model, Docker가 필요하지 않습니다.

```bash
git clone https://github.com/iamdbstjd/OSS_2026.git
cd OSS_2026

uv sync --frozen
uv run ragplan demo-plan \
  --query "Who founded Acme and who acquired it?" \
  --budget-ms 100 \
  --entity-count 2 \
  --pretty
```

출력에서 선택된 plan, 후보별 예상 p95, 제외 사유와 실제 검색 실행 여부를 확인할 수 있습니다.

```json
{
  "mode": "planner_only_no_embedding",
  "executes_retrieval": false,
  "decision": {
    "selected_plan_id": "P1",
    "effective_mode": "vector",
    "fallback_reason": "graph_audit_gate:human review and adjudication are incomplete"
  }
}
```

`demo-plan`은 embedding이나 검색 결과를 꾸며내지 않습니다. 가벼운 query feature만 분석하며
`executes_retrieval=false`를 명시합니다.

### 실제 Vector 검색

Qdrant 하나만 시작하면 checksum으로 고정된 MiniLM을 준비하고, 포함된 sample corpus를
적재한 다음 production vector engine으로 검색합니다.

```bash
cp .env.example .env
docker compose up -d qdrant
uv run ragplan quickstart-vector --pretty
```

이 경로는 다음 작업을 한 명령으로 수행합니다.

1. 허용된 `all-MiniLM-L6-v2` revision 다운로드
2. 모델 파일별 SHA-256 검증
3. sample corpus의 결정론적 chunking과 Qdrant ingest
4. canonical ID·count·schema 검증
5. 실제 vector search와 redacted trace 반환

Quickstart의 sample-v2는 원문 대소문자를 보존하는
`token-window-220-overlap-40-v2`를 명시적으로 기록합니다. 기존 benchmark와 active corpus의
`token-window-220-overlap-40-v1`은 token-ID decode 계약과 checksum을 그대로 유지합니다.

### 설치 검증

```bash
uv run ragplan verify --pretty
uv run ragplan qa --level smoke --pretty
```

실행 범위에 따라 QA 수준을 높일 수 있습니다.

```bash
# Qdrant + MiniLM + sample ingest/search
uv run ragplan qa --level vector --pretty

# 활성화된 dual-store API
uv run ragplan qa \
  --level full \
  --api-url http://127.0.0.1:8000 \
  --pretty
```

모든 QA report는 `held_out_test_accessed=false`를 기록합니다.

## 작동 방식

### 1. Analyze

`qf_v1` analyzer가 token 수, entity 밀도, 관계·비교·multi-hop·aggregation signal을 한 번만
계산합니다. Service trace에는 raw query와 embedding을 남기지 않습니다.

### 2. Plan

Rule planner는 남은 예산, 정적 p95 profile, graph audit와 circuit 상태를 이용해 P0–P8
카탈로그에서 실행 가능한 plan을 고릅니다. 선택값과 예측값은 immutable plan definition과
분리됩니다.

### 3. Execute

Vector와 Graph branch는 같은 absolute deadline 아래에서 병렬 실행됩니다. Scheduler는
응답 조립을 위해 `min(20ms, max(5ms, budget × 5%))`의 finalization reserve를 확보합니다.

### 4. Fuse and explain

Hybrid 결과는 canonical chunk ID로 중복을 제거하고 `weighted_rrf_v1`으로 융합합니다.
최종 결과에는 source별 rank, score, contribution, graph path와 fallback 사유가 남습니다.

## 검색 모드

| Planner | 동작 | 현재 상태 |
|---|---|---|
| `vector` | MiniLM + Qdrant 의미 검색 | 사용 가능 |
| `graph` | Neo4j 1–3 hop bounded traversal | 활성 corpus에서 명시적 비교 mode로 사용 가능 |
| `fixed_hybrid` | Vector·Graph 병렬 실행 + Weighted RRF | 활성 dual-store에서 사용 가능 |
| `rule` | 질의·예산·backend 상태에 따른 규칙 기반 선택 | 기본 online planner |
| `cost_aware` | 학습된 quality·latency model 기반 선택 | `research_only`, public API 비활성 |

> [!IMPORTANT]
> 100문장 human graph audit이 아직 완료되지 않아 `graph_tier_enabled=false`입니다.
> 따라서 기본 Rule planner는 안전하게 vector-only로 동작합니다. Graph와 Hybrid는 검증된
> active corpus에서 명시적으로 요청할 수 있습니다.

## CLI

| 명령 | 용도 | 필요한 인프라 |
|---|---|---|
| `ragplan demo-plan` | 검색 없이 Rule 결정 설명 | Python만 |
| `ragplan download-model` | pinned MiniLM 다운로드·checksum 검증 | 네트워크 |
| `ragplan quickstart-vector` | sample ingest부터 실제 vector search까지 | Qdrant |
| `ragplan ingest` | 사용자 corpus를 idempotent하게 Qdrant에 staging | Qdrant + MiniLM |
| `ragplan search` | 로컬 runtime 또는 REST API 검색 | 구성된 runtime |
| `ragplan verify` | package·config·선택적 live dependency 검증 | 선택 사항 |
| `ragplan qa` | `smoke`, `vector`, `full` 수준별 QA | 수준별 상이 |
| `ragplan benchmark` | Stage 9 baseline과 Stage 10 profiler | 전용 dual-store 환경 |

```bash
uv run ragplan --help
uv run ragplan search --help
```

## 필요한 인프라

처음부터 Qdrant와 Neo4j를 모두 설치할 필요는 없습니다.

| 하고 싶은 일 | 필요한 구성 |
|---|---|
| 코드와 planner 맛보기 | Python 3.12 + `uv` |
| 실제 vector demo | Qdrant + checksum-pinned MiniLM |
| Full vector·graph·hybrid | Qdrant + Neo4j + active corpus |
| 새 graph corpus 생성 | 위 구성 + pinned spaCy `en_core_web_sm` |
| 인증·멀티테넌시·agent loop | 현재 비지원 |

### Full dual-store 준비

Full runtime은 두 저장소가 존재한다는 이유만으로 활성화되지 않습니다.

1. `ragplan ingest`로 Qdrant corpus와 vector stage manifest 생성
2. `scripts/ingest_graph.py`로 동일 chunk를 Neo4j에 inactive ingest
3. `scripts/activate_corpus.py`로 count와 canonical-ID checksum reconciliation
4. 검증 성공 시에만 active corpus pointer 교체
5. `.env.example`의 Stage 6 runtime 변수를 설정하고 API 시작

Activation의 `--chunker-version`은 vector stage manifest와 정확히 같아야 하며, v2 corpus를
v1 evidence로 표시하려는 시도는 거부됩니다.

```bash
cp .env.example .env
# .env에 Stage 6 runtime 변수와 demo가 아닌 Neo4j 비밀번호를 먼저 설정하세요.
docker compose up -d --build
uv run ragplan verify --configured-runtime --pretty
```

부분 구성, model checksum 불일치 또는 Qdrant·Neo4j ID 불일치는 fail-closed로 거부됩니다.
세부 평가 계약은 [benchmark 문서](docs/benchmark.md), 실행 인자는 각 CLI와
`scripts/ingest_graph.py --help`, `scripts/activate_corpus.py --help`에서 확인할 수 있습니다.

## REST API

```bash
curl --fail http://127.0.0.1:8000/health
curl --silent --show-error http://127.0.0.1:8000/ready
curl --silent --show-error http://127.0.0.1:8000/metrics
```

| Endpoint | 의미 |
|---|---|
| `GET /health` | process liveness |
| `GET /ready` | corpus와 backend capability; Neo4j 장애 시 `degraded` |
| `GET /metrics` | request, result, error, planner, latency와 trace writer 지표 |
| `POST /v1/search` | strict `SearchRequest` 기반 검색 |

구성된 API에는 CLI를 그대로 연결할 수 있습니다.

```bash
uv run ragplan search \
  --api-url http://127.0.0.1:8000 \
  --query "What did Ada Lovelace write about?" \
  --planner rule \
  --budget-ms 500 \
  --top-k 3 \
  --pretty
```

## LLM에 근거 전달하기

RAGPlan은 answer generation을 소유하지 않습니다. [generic LLM handoff 예제](examples/llm_handoff.py)는
`SearchResponse`의 ranked chunk를 provider-neutral message로 변환하며 특정 LLM SDK를
import하지 않습니다.

```bash
uv run ragplan search \
  --query "What did Ada Lovelace write about?" \
  --planner vector \
  --pretty > /tmp/ragplan-response.json

uv run python examples/llm_handoff.py \
  --response /tmp/ragplan-response.json \
  --question "What did Ada Lovelace write about?"
```

## 장애 대응과 privacy

- 최대 32개 in-flight request, queue 없는 admission control
- backend별 연속 5회 실패 시 30초 circuit open, half-open probe 1개
- 요청 내부 retry 없음
- 한 branch 실패 시 성공 branch 결과를 보존하는 typed partial response
- client disconnect 시 모든 child task cancel·await
- `RAGPLAN_FORCE_VECTOR_ONLY`와 `RAGPLAN_DISABLE_COST_AWARE` kill switch
- parameterized Cypher와 32KiB request 제한
- raw query, embedding, 전체 문서와 임의 metadata를 제외한 redacted trace
- 10MiB 단위 회전, 현재 파일 포함 최대 5개의 trace file
- trace queue overflow와 filesystem 오류를 검색 실패로 전파하지 않음

Service mode는 `RAGPLAN_LOGGING__MODE=redacted`만 허용하고 다른 모든 값을 startup에서
거부합니다. Drop과 write failure는 `/metrics`의 `trace_dropped_count`,
`trace_write_failure_count`에 반영됩니다.

## 검증 결과

RAGPlan은 성공한 결과만 골라서 보고하지 않습니다. Timeout, backend error, partial과 zero-hit도
raw evidence와 평가 분모에 유지합니다.

| Evidence | 규모·결과 |
|---|---|
| Frozen benchmark | 600 query, train/validation/test = 360/120/120 group split |
| Active dual-store corpus | Qdrant와 Neo4j 각각 8,604 canonical chunk |
| Stage 9 baseline | 224,640 raw trial rows |
| Stage 10 profiler | 199,680 raw trial rows |
| 전체 실행 기록 | 424,320 rows |
| Training matrix | 15,360 rows |
| Oracle@Budget | 1,920 labels |
| Quality model | validation MAE 0.009219 |
| Latency model | overall p95 coverage 0.928368 |

학습 모델은 일부 지표를 통과했지만 plan-pair ranking, plan별 latency coverage와 pinball
improvement gate를 모두 통과하지 못했습니다. Stage 12 runtime guard도 p95 underprediction
rate 0.21에서 모델을 비활성화했습니다.

따라서 현재 상태는 다음과 같습니다.

```text
Online default      rule
Rule graph routing  disabled until human audit
Cost-aware serving  disabled
Cost-aware status   research_only / offline comparison only
```

실패한 모델을 숨기지 않고 public serving을 차단한 것이 RAGPlan의 MLOps 안전 계약입니다.

## 대상 사용자와 범위

### 이런 팀에 적합합니다

- latency SLO 안에서 vector·graph 전략을 조정하는 RAG 개발팀
- timeout, circuit breaker와 partial result가 필요한 AI infrastructure 팀
- query×plan×budget matrix와 Oracle@Budget을 재현하는 연구자
- 생성 모델과 독립된 ranked evidence layer가 필요한 서비스

### 현재 제공하지 않습니다

- 완성형 질문답변 UI 또는 LLM answer generation
- 인증, 멀티테넌시와 분산 production control plane
- agent loop와 tool orchestration
- 사람 검토 없이 자동 graph 품질을 보장한다는 주장
- `research_only` cost model의 public serving

## 개발

### 요구 사항

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker Engine + Docker Compose v2 — integration 작업에만 필요

### 품질 검사

```bash
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not integration and not e2e"
```

실제 backend 통합 test는 환경이 준비된 경우에만 실행합니다.

```bash
MODEL_SNAPSHOT="models/minilm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"

RAGPLAN_TEST_QDRANT_URL=http://127.0.0.1:6333 \
RAGPLAN_TEST_MODEL_SNAPSHOT="$MODEL_SNAPSHOT" \
uv run pytest -q tests/integration/test_qdrant_vector_backend.py

RAGPLAN_TEST_NEO4J_URI=bolt://127.0.0.1:7687 \
RAGPLAN_TEST_NEO4J_USER=neo4j \
RAGPLAN_TEST_NEO4J_PASSWORD=ragplan-demo-change-me \
uv run pytest -q tests/integration/test_graph_retrieval.py
```

## 문서

- [Graph·Hybrid·Scheduler·Rule runtime 운영](docs/runtime.md)
- [Benchmark protocol과 Stage 9·10 evidence](docs/benchmark.md)
- [Stage 11 model training과 validation gate](docs/model_training.md)
- [Stage 12 offline cost policy](docs/offline_cost_policy.md)
- [LLM handoff와 활용 예제](examples/README.md)
- [제3자 소프트웨어·모델·데이터 라이선스](THIRD_PARTY_LICENSES.md)
- [보안 정책](SECURITY.md)

## 기여하기

Issue와 pull request를 환영합니다. 변경하기 전에 [CONTRIBUTING.md](CONTRIBUTING.md)의
개발 환경, test, privacy와 artifact 계약을 확인해 주세요.

```bash
git clone https://github.com/iamdbstjd/OSS_2026.git
cd OSS_2026
uv sync --frozen
uv run pytest -m "not integration and not e2e"
```

버그를 공개 Issue로 보고하기 곤란한 경우 [Security Policy](SECURITY.md)의 private advisory
절차를 이용해 주세요.

## 라이선스

RAGPlan의 자체 코드는 [Apache License 2.0](LICENSE)으로 배포됩니다.

Qdrant, Neo4j, Sentence Transformers, MiniLM, spaCy와 benchmark dataset은 각각의 원래
라이선스를 유지합니다. 정확한 revision, image digest, checksum과 attribution은
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)에 기록되어 있습니다.

---

<p align="center">
  Built by <strong>ProSheet</strong> · Evidence before claims · Fail closed, recover gracefully
</p>
