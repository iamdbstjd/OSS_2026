# RAGPlan Implementation Stages

> 상태: Implementation-ready plan  
> 기준일: 2026-08-09  
> 요구사항 기준: [`adaptive_rag_query_optimizer_PRD.md`](./adaptive_rag_query_optimizer_PRD.md)  
> 확정 결정: [`adaptive_rag_query_optimizer_PRD_addendum.md`](./adaptive_rag_query_optimizer_PRD_addendum.md)  
> 범위: 문서·설계 계획이며 이 파일 자체는 코드 구현을 포함하지 않는다.

## 1. 이 문서의 사용법

이 문서는 단순 일정표가 아니라 Stage gate 문서다. 각 Stage는 다음 순서로 진행한다.

```text
Entry Criteria 확인
→ 구현
→ 해당 Stage의 unit/integration test 실행
→ 산출물과 trace 확인
→ Exit Criteria 충족
→ 다음 Stage 시작
```

Exit Criteria를 통과하지 못한 상태에서 downstream Stage를 시작하지 않는다. 특히 다음 규칙은 예외가 없다.

```text
Final scheduler/deadline semantics가 동결되기 전
→ offline profiler 실행 금지
→ latency model 학습 금지
→ cost-aware 성능 수치 발표 금지
```

## 2. 목표와 완료선

### Competition survival line

다음 기능이 완성되면 adaptive ML이 실패해도 정직하게 제출 가능한 baseline이 된다.

```text
Vector-only
Graph-only
Fixed Hybrid
Rule Planner
Deadline-aware Scheduler/Fallback
Frozen benchmark
REST/CLI
Docker reproduction
```

해당 범위는 Stage 0~9와 Stage 13의 필수 항목이다.

### Differentiated target line

다음 기능까지 완료되어야 제품의 핵심 차별화 주장을 할 수 있다.

```text
Query × Plan Profiler
Quality/Latency Model
Cost-aware Optimizer
Oracle/BestFixed 비교
Pareto success gate
```

해당 범위는 Stage 10~14다.

## 3. 공통 개발 원칙

1. API, CLI, benchmark, profiler는 동일한 `RAGPlanEngine.search()` 실행 경로를 사용한다.
2. Offline benchmark만을 위한 별도 retrieval 구현을 만들지 않는다.
3. 모든 ID, plan, trace, dataset, model, config는 version/hash를 가진다.
4. latency는 monotonic clock으로 측정한다.
5. raw query와 embedding은 기본 trace에 저장하지 않는다.
6. Qdrant와 Neo4j의 partial ingestion을 정상 상태로 간주하지 않는다.
7. test split으로 threshold, plan, model, feature를 조정하지 않는다.
8. P0에 새로운 framework, DB adapter, LLM router, reranker를 추가하지 않는다.
9. 구현 완료 주장은 fresh test output과 재현 artifact로 증명한다.
10. 한 PR은 하나의 명확한 Stage task 또는 하나의 교차 계약 변경만 포함한다.

## 4. 전체 Stage 맵

| Stage | 이름 | 핵심 산출물 | 상대 난이도 | 선행 Stage |
|---:|---|---|---|---|
| 0 | Repository & Reproducibility Bootstrap | 실행 가능한 package, Compose, CI, license skeleton | M | 없음 |
| 1 | Core Contracts & Deadline Foundation | validated domain model, plan catalog, deadline/error contract | L | 0 |
| 2 | Benchmark Data & Qrels Foundation | frozen 600-query manifest, splits, qrels, metric fixtures | L | 0, 1; qrels는 3의 S3-001 |
| 3 | Vector Vertical Slice | chunk→embed→Qdrant→search→trace | M | 0, 1 |
| 4 | Graph Ingestion | entity/relation graph, versioned dual-write, reconcile | XL | 1, 2, 3 |
| 5 | Graph Retrieval | bounded traversal, path/chunk recovery, graph ranking | L | 4 |
| 6 | Fixed Hybrid & Fusion | deterministic weighted RRF, provenance-preserving hits | M | 3, 5 |
| 7 | Scheduler, Deadline & Fallback | final runtime semantics, parallel execution, partial results | XL | 1, 6 |
| 8 | Query Analyzer & Rule Planner | QueryAnalysis, feature schema, deterministic planner | L | 1, 7 |
| 9 | Benchmark Harness V1 | comparable baseline results, BestFixed, warm/cold protocol | XL | 2, 7, 8 |
| 10 | Offline Plan Profiler | query×plan raw trials, Oracle@Budget | L | 9 runtime freeze |
| 11 | Cost Models & Artifact Lifecycle | quality/p95 latency models, compatible model artifact | XL | 10 |
| 12 | Cost-aware Optimizer | budget-feasible plan selection and explanation | L | 11 |
| 13 | API, CLI, Observability & Packaging | public surface, readiness, trace, examples, Docker image | L | 7, 8; 12 optional |
| 14 | Final Evidence & Submission Freeze | test results, Pareto/ablation, clean-machine proof | XL | 9, 12, 13 |
| 15 | Post-MVP Options | reranker, Korean, Prometheus, extra adapters | 별도 | 14 이후 |

### Critical path

```text
S0 → S1 → S3 → S4 → S5 → S6 → S7
                                  ↓
S2 ─────────────────────────────→ S9 → S10 → S11 → S12 → S14
                                  ↑                   ↑
                              S8 ─┘              S13 ─┘
```

S2의 license/download/query-selection 작업은 S3와 병렬 진행할 수 있다. 다만 S2의 qrels는 S3-001의 최종 chunker를 사용해야 하므로 Stage 2 전체 Exit는 S3-001 이후에만 가능하다. S13의 기본 API/CLI shell은 일찍 만들 수 있으나 public contract 확정은 S7 이후에 한다.

## 5. 목표 repository 구조

```text
ragplan/
├── README.md
├── LICENSE
├── THIRD_PARTY_LICENSES.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── uv.lock
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── .env.example
├── configs/
│   ├── default.yaml
│   ├── plans.yaml
│   └── benchmark.yaml
├── src/ragplan/
│   ├── api/
│   │   ├── server.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── errors.py
│   ├── cli/
│   │   └── app.py
│   ├── core/
│   │   ├── engine.py
│   │   ├── models.py
│   │   ├── ids.py
│   │   ├── deadline.py
│   │   ├── config.py
│   │   ├── errors.py
│   │   └── versions.py
│   ├── backends/
│   │   ├── vector/
│   │   │   ├── base.py
│   │   │   └── qdrant.py
│   │   └── graph/
│   │       ├── base.py
│   │       └── neo4j.py
│   ├── ingestion/
│   │   ├── models.py
│   │   ├── normalize.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── entities.py
│   │   ├── relations.py
│   │   ├── resolver.py
│   │   ├── manifest.py
│   │   ├── reconcile.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   ├── vector.py
│   │   ├── graph.py
│   │   ├── fusion.py
│   │   └── reranker.py
│   ├── planner/
│   │   ├── analyzer.py
│   │   ├── features.py
│   │   ├── catalog.py
│   │   ├── rule.py
│   │   ├── optimizer.py
│   │   ├── quality_model.py
│   │   ├── latency_model.py
│   │   └── artifacts.py
│   ├── scheduler/
│   │   ├── executor.py
│   │   ├── states.py
│   │   └── cancellation.py
│   └── observability/
│       ├── trace.py
│       ├── metrics.py
│       └── logging.py
├── benchmark/
│   ├── README.md
│   ├── manifests/
│   ├── datasets/
│   ├── qrels/
│   ├── configs/
│   ├── runners/
│   ├── metrics/
│   ├── analysis/
│   └── results/
├── scripts/
│   ├── ingest.py
│   ├── search.py
│   ├── benchmark.py
│   ├── profile_plans.py
│   ├── train_cost_models.py
│   └── verify_reproduction.py
├── examples/
└── tests/
    ├── fixtures/
    ├── unit/
    ├── contract/
    ├── integration/
    ├── e2e/
    └── benchmark/
```

`benchmark/results/`의 대용량 raw 결과와 model artifact는 기본적으로 Git에 commit하지 않는다. 제출에 필요한 frozen summary, manifest, checksum, 작은 sample artifact만 별도 정책으로 포함한다.

---

# Stage 0 — Repository & Reproducibility Bootstrap

## 목표

아무 기능도 없는 상태에서도 동일한 Python 환경과 Qdrant/Neo4j 서비스를 모든 개발자가 재현할 수 있게 한다.

## Entry Criteria

- Addendum 결정이 승인 상태
- 공개 working name과 package import 이름이 `RAGPlan` / `ragplan`으로 결정됨

## 구현 작업

### S0-001 Package bootstrap

- `src/ragplan` layout 생성
- Python 3.12 requirement 설정
- PEP 621 metadata 작성
- console entry point `ragplan` 등록
- 최소 `ragplan --version` 명령 제공

### S0-002 Dependency groups

- runtime, development, benchmark, graph-extraction dependency group 분리
- `uv.lock` 생성
- `pip install -e .` 경로도 유지
- unpinned direct URL dependency 금지

### S0-003 Docker Compose

- Qdrant, Neo4j, API service 정의
- named volume과 명시적 port 설정
- service healthcheck 작성
- `latest` tag 금지
- clean-machine 검증 후 image digest 기록

### S0-004 Configuration skeleton

- `.env.example`
- `configs/default.yaml`
- environment variable override 규칙
- secret 필드는 파일에 기본값을 넣지 않음
- sample Neo4j password는 demo 전용임을 명시

### S0-005 Quality tooling

- Ruff lint/format
- mypy
- pytest
- pytest-asyncio
- unit/integration/e2e marker
- CI에서 unit test와 import smoke 실행

### S0-006 OSS files

- Apache-2.0 `LICENSE`
- `THIRD_PARTY_LICENSES.md` skeleton
- `CONTRIBUTING.md`
- `SECURITY.md`
- dependency/dataset/model license 표 형식 정의

## 주요 파일

```text
pyproject.toml
uv.lock
docker-compose.yml
Dockerfile
.env.example
configs/default.yaml
.github/workflows/ci.yml
LICENSE
THIRD_PARTY_LICENSES.md
```

## 검증

```bash
uv sync --frozen
uv run ragplan --version
docker compose config
docker compose up -d
uv run pytest -m "not integration and not e2e"
uv run ruff check .
uv run mypy src
```

## Exit Criteria

- [ ] Fresh checkout에서 package import 성공
- [ ] Qdrant/Neo4j healthcheck 성공
- [ ] CI에서 lint/typecheck/unit smoke 통과
- [ ] repository에 secret 없음
- [ ] dependency와 image version이 명시됨

## 금지 사항

- Retrieval 구현 시작 금지
- `latest` Docker image 사용 금지
- 모델 weight나 benchmark dataset을 임의로 repository에 commit 금지

---

# Stage 1 — Core Contracts & Deadline Foundation

## 목표

모든 downstream component가 공유할 domain contract, ID, plan invariant, deadline, error schema를 먼저 고정한다.

## Entry Criteria

- Stage 0 완료
- Addendum의 plan catalog와 deadline 정의 확인

## 구현 작업

### S1-001 Domain models

다음 immutable/validated model을 구현한다.

```text
Chunk
Entity
Relation
RetrievalHit
GraphPath
QueryFeatures
QueryAnalysis
PlanDefinition
PlanEstimate
PlannerDecision
BranchResult
SearchRequest
SearchResponse
SearchTrace
IngestionManifest
ModelManifest
```

`QueryFeatures`와 `SearchTrace`에는 raw embedding을 넣지 않는다.

### S1-002 Plan catalog

- P0~P8를 `configs/plans.yaml`과 typed loader로 정의
- P7은 기본 P0 plan space에서 비활성
- plan ID 중복 검사
- weight/depth/top-k/rerank invariant 검사
- plan catalog canonical serialization과 SHA256 생성

### S1-003 Request validation

- query: trim 후 1~4096 Unicode code point
- request body 최대 32 KiB, unknown field 거부
- latency budget: 25~5000 ms
- top-k: 1~50
- `cost_aware`는 top-k 10만 허용
- planner mode enum 고정

### S1-004 Deadline abstraction

- monotonic clock interface
- absolute deadline
- elapsed/remaining 계산
- finalization reserve 계산
- fake clock 주입 가능
- system wall clock을 timeout 계산에 사용하지 않음

### S1-005 Error taxonomy

Stable internal/API error code를 정의한다.

```text
INVALID_REQUEST
INVALID_QUERY
PLAN_INVARIANT_VIOLATION
NOT_READY
DEPENDENCY_UNAVAILABLE
MODE_UNAVAILABLE
OVERLOADED
DEADLINE_EXCEEDED
MODEL_INCOMPATIBLE
CORPUS_INCONSISTENT
RETRIEVAL_FAILED
INTERNAL_ERROR
```

### S1-006 Backend protocols

검색뿐 아니라 lifecycle에 필요한 최소 contract를 정의한다.

```text
VectorBackend.search(embedding, top_k, corpus_version, deadline)
GraphBackend.search(query_analysis, plan, corpus_version, deadline)
Backend.health()
Backend.close()
```

Ingestion writer interface는 runtime search interface와 분리한다.

## 주요 파일

```text
src/ragplan/core/models.py
src/ragplan/core/ids.py
src/ragplan/core/deadline.py
src/ragplan/core/errors.py
src/ragplan/core/config.py
src/ragplan/planner/catalog.py
src/ragplan/backends/vector/base.py
src/ragplan/backends/graph/base.py
configs/plans.yaml
tests/unit/core/
tests/contract/
```

## 테스트

- 모든 plan invariant의 valid/invalid table test
- top-k/budget/query boundary test
- fake clock으로 remaining/reserve test
- stable serialization/hash test
- error code와 HTTP mapping snapshot test
- embedding이 trace serialization에 포함되지 않는지 test

## Exit Criteria

- [ ] 모든 plan preset이 완전한 `PlanDefinition`으로 load됨
- [ ] invalid plan은 startup에서 즉시 실패
- [ ] deadline test가 sleep 없이 deterministic하게 통과
- [ ] API schema와 internal model의 필드 의미가 문서와 일치
- [ ] downstream 구현이 임의 dict 대신 domain model을 사용할 수 있음

---

# Stage 2 — Benchmark Data & Qrels Foundation

## 목표

코드를 최적화하기 전에 무엇을 정답으로 간주하고 어떻게 비교할지 동결한다.

## Entry Criteria

- Stage 0 완료
- Stage 1의 canonical ID 함수 확정
- upstream dataset license 확인을 시작할 수 있음
- S2-006을 시작하기 전 S3-001 chunker contract와 golden fixture 확정

## 구현 작업

### S2-001 Dataset license audit

Natural Questions/DPR, HotpotQA, MuSiQue 각각에 대해 다음을 기록한다. 기준 라이선스는 각각 `CC BY-SA 3.0`, `CC BY-SA 4.0`, `CC BY 4.0`이며 upstream 공식 URL과 실제 다운로드 URL을 함께 보존한다.

```text
official download URL
upstream version/date
license/terms URL
redistribution 가능 여부
required attribution
raw archive SHA256
```

License가 명확하지 않은 raw artifact는 commit하지 않는다.

### S2-002 Download/preprocess commands

- source별 download adapter
- checksum 검증
- interrupted download recovery
- local cache 경로
- upstream format을 internal normalized format으로 변환

### S2-003 Frozen query selection

- NQ 200, Hotpot bridge 200, Hotpot comparison 100, MuSiQue 100 선택
- query ID 목록을 manifest에 고정
- 선택 seed와 filtering rule 저장
- source별 train split만 사용하고 hidden-label test split은 사용하지 않음
- `SHA256(source_dataset + ":" + source_query_id + ":20260809)` 오름차순 선택
- missing/duplicate query 제거 보고서 생성
- MuSiQue 공개 dev/test single-hop source ID와 NQ 선택분의 교집합 0 검증
- ID 비교가 불가능한 경우 normalized question exact/hash overlap 0 검증
- MuSiQue는 decomposition step이 정확히 3개인 record만 사용

### S2-004 Corpus construction

- 선택 query의 supporting/distractor passage만 수집
- duplicate document와 near-duplicate passage 식별
- canonical `document_id` 부여
- source attribution 보존

### S2-005 Split generation

- 360/120/120 group split
- source별 quota: NQ 120/40/40, Hotpot bridge 120/40/40, Hotpot comparison 60/20/20, MuSiQue 60/20/20
- document/entity cluster leakage 검사
- query template duplicate 검사
- taxonomy는 multi-label로 저장하고 test의 각 필수 tag를 최소 15개 보장
- tag rule은 Addendum 표를 그대로 구현하고 tag 판정 근거를 manifest에 저장
- group 제약과 quota가 충돌하면 source pool을 확대하고 group 자체는 분할하지 않음
- split hash 저장

### S2-006 Qrels generation

- supporting sentence/paragraph을 chunk와 연결
- S3-001의 production chunker를 import하여 사용하고 benchmark 전용 chunker를 만들지 않음
- grade 1/2 생성
- orphan query와 orphan qrel 거부
- 한 query에 relevant chunk가 0개면 benchmark에서 제외하고 이유 기록

### S2-007 Metric library

다음을 pure function으로 구현한다.

```text
Recall@5
Recall@10
MRR@10
nDCG@10
```

### S2-008 Synthetic graph fixture

- 1/2/3-hop deterministic graph 생성
- exact expected entity path와 relevant chunk 정의
- cycle, hub, disconnected entity fixture 포함

## 주요 파일

```text
benchmark/manifests/adaptive_rag_bench_v1.yaml
benchmark/manifests/licenses.yaml
benchmark/qrels/qrels_v1.jsonl
benchmark/configs/splits_v1.json
benchmark/datasets/.gitkeep
benchmark/metrics/
tests/fixtures/benchmark/
tests/benchmark/test_metrics.py
```

## 테스트

- checksum mismatch failure
- same seed produces same query list/split
- no query crosses split
- no query×plan row split leakage
- qrels referential integrity
- hand-calculated metric fixtures
- graded nDCG fixture
- zero-relevant query validation failure

## Exit Criteria

- [ ] 정확히 600 query manifest 생성
- [ ] train/validation/test가 360/120/120
- [ ] source quota와 필수 query-tag 최소 수 충족
- [ ] 모든 query에 최소 1 relevant chunk 존재
- [ ] license/attribution 문서화 완료
- [ ] metric fixture와 reference 계산이 일치
- [ ] test split query ID는 별도 immutable manifest로 freeze

## Stop Condition

Dataset license 또는 qrels mapping이 해결되지 않으면 Stage 9 이후로 진행하지 않는다. 임의 synthetic data로 primary 결과를 대체하지 않는다.

---

# Stage 3 — Vector Vertical Slice

## 목표

하나의 document가 deterministic chunk와 embedding으로 Qdrant에 저장되고 동일 engine path를 통해 검색되도록 한다.

## Entry Criteria

- Stage 1 완료
- embedding model revision `b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`과 Apache-2.0 license 확인
- Qdrant image/client version freeze 후보 존재

## 구현 작업

### S3-001 Text normalization/chunking

- Unicode normalization
- 220-token chunk, 40-token overlap
- position과 content hash 생성
- empty/duplicate chunk 제거
- canonical chunk ID 생성

### S3-002 Embedder

- `all-MiniLM-L6-v2`를 revision `b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`으로 local loading
- revision/checksum validation
- batch document embedding
- single query embedding
- dimension 384 assertion
- normalized vector policy 고정

### S3-003 Qdrant collection management

- cosine/384 collection 생성
- payload schema 정의
- `corpus_version`, `canonical_chunk_id`, `document_id`, `position`, `text` 저장
- 필요한 payload index 생성
- collection/schema mismatch 시 fail fast

### S3-004 Vector writer

- Qdrant UUIDv5 point ID
- batch upsert
- idempotent same-version write
- write count/checksum 반환

### S3-005 Vector backend

- async Qdrant client 사용
- query embedding 입력
- corpus version filter
- top-k validation
- deadline/timeout 전달
- Qdrant point를 `RetrievalHit`으로 변환

### S3-006 Engine vector mode

- `planner=vector` 경로
- analysis/embedding/vector/total trace
- final top-k 적용
- zero-hit 정상 응답

## 주요 파일

```text
src/ragplan/ingestion/normalize.py
src/ragplan/ingestion/chunker.py
src/ragplan/ingestion/embedder.py
src/ragplan/backends/vector/qdrant.py
src/ragplan/retrieval/vector.py
src/ragplan/core/engine.py
scripts/ingest.py
scripts/search.py
tests/unit/ingestion/
tests/integration/test_qdrant.py
```

## 테스트

- token boundary와 overlap
- content 변경 시 chunk ID 변경
- 동일 input 재수집 시 point 증가 없음
- invalid embedding dimension
- empty/nonexistent collection
- corpus version isolation
- query embedding 한 번만 생성됨
- Qdrant 반환 순서와 hit mapping

## Exit Criteria

- [ ] sample corpus ingest 성공
- [ ] vector query가 relevant chunk 반환
- [ ] Qdrant ID는 UUID, payload에는 canonical string ID 존재
- [ ] embedding/Qdrant/total latency trace 존재
- [ ] integration test가 real Docker Qdrant에서 통과

---

# Stage 4 — Graph Ingestion

## 목표

Vector와 같은 chunk를 사용해 deterministic entity/relation graph를 만들고, partial dual-write를 활성 corpus로 노출하지 않는다.

## Entry Criteria

- Stage 1 완료
- Stage 2 normalized corpus format 존재
- Stage 3 canonical document/chunk ID와 active corpus version 규칙 동작

## 구현 작업

### S4-001 Entity extraction

- pinned spaCy pipeline 로드
- 허용 entity type filtering
- raw mention, normalized name, sentence span 보존
- unsupported/no-entity chunk 처리

### S4-002 Entity resolution V0

- exact normalized alias match
- entity UUIDv5 생성
- 동일 이름/다른 type 분리
- alias/source mention provenance 저장

### S4-003 Relation extraction V0

- direct SVO
- passive voice
- copular relation
- appositional relation
- predicate lemma normalization
- rule tier별 confidence
- confidence <0.70 drop

### S4-004 Neo4j schema

- uniqueness constraints: Document, Chunk, Entity IDs
- indexes: corpus version, normalized entity name/type
- Document→Chunk `HAS_CHUNK`
- Chunk→Entity `MENTIONS`
- Entity→Entity `RELATES_TO`
- relation provenance property

### S4-005 Neo4j writer

- parameterized Cypher만 사용
- batch write
- same-version idempotency
- partial batch failure 기록
- transaction timeout 설정

### S4-006 Ingestion manifest

다음을 저장한다.

```text
ingestion_run_id
corpus_version
source checksum
chunker version
embedding model revision
extractor version
Qdrant count/checksum/status
Neo4j count/checksum/status
activation status
```

### S4-007 Reconciliation and activation

- Qdrant/Neo4j canonical chunk ID set 비교
- count/checksum 불일치 시 activation 거부
- 이전 active corpus 유지
- 성공 시 atomic local active-version pointer 전환
- failed run 재시도 또는 폐기 명령

### S4-008 Extraction quality audit

- train passage에서 SHA256 순으로 100 sentence 고정
- entity span/type과 directed relation을 사람이 검토
- 20 sentence는 두 reviewer가 독립 annotation 후 adjudication
- entity micro precision/recall/F1, relation precision/recall, reviewer agreement 기록
- entity F1 < 0.75 또는 relation precision < 0.70이면 rule planner graph tier 비활성화
- audit sheet/version/checksum을 benchmark manifest에 연결

## 주요 파일

```text
src/ragplan/ingestion/entities.py
src/ragplan/ingestion/relations.py
src/ragplan/ingestion/resolver.py
src/ragplan/ingestion/manifest.py
src/ragplan/ingestion/reconcile.py
src/ragplan/ingestion/pipeline.py
src/ragplan/backends/graph/neo4j.py
tests/unit/ingestion/test_entities.py
tests/unit/ingestion/test_relations.py
tests/integration/test_neo4j_ingestion.py
tests/integration/test_reconcile.py
benchmark/audits/graph_extraction_v1/
```

## 테스트

- case/whitespace/punctuation normalization
- same name, different entity type
- active/passive/copular/appositional extraction fixtures
- unsupported sentence produces no false relation
- Cypher parameters used; raw text interpolation 없음
- repeated ingestion is idempotent
- Qdrant success/Neo4j failure prevents activation
- Neo4j success/Qdrant failure prevents activation
- new version activation and old version rollback
- audit sample selection 재현성과 metric 계산 fixture

## Exit Criteria

- [ ] fixture document의 expected entity/relation 정확히 생성
- [ ] 모든 relation에 source chunk와 extractor version 존재
- [ ] Qdrant/Neo4j chunk ID reconciliation 성공
- [ ] partial write 상태에서 search가 새 version을 보지 않음
- [ ] re-ingest가 duplicate node/edge를 만들지 않음
- [ ] graph extraction audit와 reviewer agreement 기록 완료
- [ ] audit gate 실패 시 rule graph tier가 disabled 상태로 전달됨

## 주요 위험

Graph extraction 품질이 낮아도 Stage 자체를 숨기지 않는다. Fixture correctness와 extraction audit sample을 별도로 기록하고, benchmark 품질은 Stage 9에서 판단한다.

---

# Stage 5 — Graph Retrieval

## 목표

Query seed entity에서 bounded relationship traversal을 수행하고, 설명 가능한 path와 관련 chunk를 안정적으로 반환한다.

## Entry Criteria

- Stage 4 complete/active corpus 존재
- synthetic graph fixture와 path qrels 준비

## 구현 작업

### S5-001 Query seed extraction

- ingestion과 동일 normalization/entity pipeline 재사용
- 최대 seed 5개
- exact normalized alias lookup
- seed lookup score와 unmatched seed 기록

### S5-002 Bounded traversal

- `RELATES_TO`만 탐색
- 양방향 discovery, edge direction 보존
- plan depth 1~3
- repeated-node path 제거
- max path/entity bounds 적용
- Neo4j transaction timeout과 remaining deadline 연동

### S5-003 Chunk recovery

- reached entity ← `MENTIONS` ← Chunk로 별도 recovery
- active corpus version filter
- 최대 100 candidate
- 동일 chunk의 여러 path 보존 또는 capped aggregation

### S5-004 Graph scoring

- Addendum의 V0 formula 구현
- seed overlap/hop/confidence contribution 기록
- deterministic tie-break
- final graph top-k 적용

### S5-005 Graph trace

- seed entities/matches
- requested/actual depth
- visited entity/path count
- limit hit 여부
- query/recovery/ranking latency
- timeout/error status

## 주요 파일

```text
src/ragplan/retrieval/graph.py
src/ragplan/backends/graph/neo4j.py
src/ragplan/core/models.py
tests/unit/retrieval/test_graph_scoring.py
tests/integration/test_graph_retrieval.py
tests/benchmark/test_synthetic_paths.py
```

## 테스트

- exact 1/2/3-hop expected path
- cycle graph terminates
- hub/fan-out bounds respected
- `MENTIONS`/`HAS_CHUNK`가 relationship traversal에 포함되지 않음
- no seed, unmatched seed, no path, no chunk cases
- relation direction preserved
- same score stable chunk-ID ordering
- graph timeout returns typed branch state

## Exit Criteria

- [ ] synthetic path Recall 100% for supported fixture
- [ ] traversal limit을 초과하지 않음
- [ ] graph-only CLI/search path 성공
- [ ] path와 chunk provenance가 response model에 유지
- [ ] `EXPLAIN`/`PROFILE` 결과를 development artifact로 저장

---

# Stage 6 — Fixed Hybrid & Fusion

## 목표

Vector와 Graph 결과를 동일 chunk identity로 deduplicate하고 deterministic cross-store fusion을 수행한다.

## Entry Criteria

- Stage 3 vector path 완료
- Stage 5 graph path 완료
- 동일 active corpus version 사용 확인

## 구현 작업

### S6-001 Common hit contract

- vector/graph hit을 canonical chunk ID로 통일
- final hit에 multiple sources 지원
- source별 original rank/score 저장
- graph path와 source metadata 유지

### S6-002 `weighted_rrf_v1`

- 1-based rank
- `k=60`
- plan weight 적용
- missing branch 처리
- deterministic tie-break

### S6-003 Deduplication

- 동일 canonical ID merge
- text/document metadata 충돌 시 consistency error 기록
- graph paths deduplicate/cap
- 동일 text이지만 ID가 다른 chunk는 자동 merge하지 않음

### S6-004 Fixed Hybrid engine path

- P4/P5/P6/P8 명시적 실행 지원
- 기본 fixed hybrid는 P5
- final request top-k 적용
- fusion contribution trace

## 주요 파일

```text
src/ragplan/retrieval/fusion.py
src/ragplan/core/engine.py
tests/unit/retrieval/test_fusion.py
tests/contract/test_hit_provenance.py
tests/integration/test_fixed_hybrid.py
```

## 테스트

- hand-calculated weighted RRF
- duplicate in both branches
- one/zero branch result
- equal score deterministic order
- weights 1/0, 0/1, 0.5/0.5
- metadata conflict
- provenance survives serialization

## Exit Criteria

- [ ] vector/graph/fixed_hybrid 세 mode 동일 engine에서 동작
- [ ] fusion 결과가 test fixture와 정확히 일치
- [ ] final hit에 source rank/contribution/path가 존재
- [ ] 동일 corpus version이 아니면 hybrid 실행 거부

---

# Stage 7 — Scheduler, Deadline & Graceful Degradation

## 목표

Serving과 profiler가 공유할 최종 실행 의미를 완성한다. 이 Stage 이후 scheduler semantics를 freeze한다.

## Entry Criteria

- Stage 1 deadline/state/error contracts 완료
- Stage 6 fixed hybrid 동작
- Qdrant/Neo4j async client 경로 사용 가능

## 구현 작업

### S7-001 Scheduler state machine

Request state:

```text
received → analyzing → planning → executing
→ fusing → reranking(optional)
→ complete | partial | failed
```

Branch state:

```text
not_scheduled → running
→ succeeded | timed_out | failed | cancelled
```

Invalid state transition은 내부 오류로 처리한다.

### S7-002 Parallel execution

- 활성 vector/graph branch만 task 생성
- 동일 event loop에서 실제 non-blocking client 사용
- 두 branch start barrier 기록
- wall-clock이 직렬 합이 아닌지 검증

### S7-003 Deadline allocation

- request absolute deadline 공유
- analyzer/planner 실제 elapsed 차감
- finalization reserve 보존
- branch별 임의 독립 budget 생성 금지

### S7-004 Backend-native timeout

- Neo4j transaction timeout 설정
- Qdrant request/transport timeout 설정
- sub-second app deadline은 scheduler가 책임
- client timeout과 app timeout을 trace에서 구분

### S7-005 Cancellation cleanup

- timeout task cancel/await
- Neo4j session context 종료
- Qdrant client response cleanup
- client disconnect propagation
- background orphan task 0개 보장

### S7-006 Graceful degradation

- vector success + graph timeout/error → partial vector
- graph success + vector timeout/error → partial graph
- both successful → fusion
- both zero-hit success → complete empty
- deadline 내 result 없음 → `DEADLINE_EXCEEDED`
- fusion 전 budget 소진 → available ranking 반환, optional work skip

### S7-007 Execution trace

- expected vs actual state
- branch start/end/duration
- cancel reason
- budget at each phase boundary
- budget violation/overshoot
- fallback reason

### S7-008 Circuit breaker와 admission control

- backend별 연속 failure counter와 `closed/open/half_open` state
- 5회 연속 backend-native transport/timeout failure 후 30초 open
- 정상적인 application deadline cancellation은 failure counter에서 제외
- half-open에서 동시 probe 1개
- request 내부 자동 retry 0회
- process in-flight request 기본 32, request당 backend task 최대 2
- admission slot을 즉시 얻지 못하면 queue하지 않고 typed `OVERLOADED` 반환
- force-vector/cost-aware-disable kill switch를 engine 진입 시 snapshot

## 주요 파일

```text
src/ragplan/scheduler/states.py
src/ragplan/scheduler/executor.py
src/ragplan/scheduler/cancellation.py
src/ragplan/core/deadline.py
src/ragplan/core/engine.py
tests/unit/scheduler/
tests/integration/test_scheduler_backends.py
tests/unit/scheduler/test_circuit_breaker.py
tests/unit/scheduler/test_admission.py
```

## 테스트

Fake delayed backend와 fake clock을 사용한다.

- vector 30ms, graph 80ms wall-clock 병렬성
- vector timeout / graph success
- graph timeout / vector success
- both timeout
- backend exception + other success
- both backend exception
- fusion/finalization reserve
- task cancellation awaited
- client disconnect
- no orphan task/connection leak
- deadline overshoot tolerance 기록
- circuit closed/open/half-open state와 recovery
- request retry가 0회인지 검증
- in-flight 32 초과 시 bounded overload
- kill switch가 graph/model path를 호출하지 않는지 검증

## Exit Criteria — Runtime Freeze Gate

- [ ] 모든 state transition test 통과
- [ ] parallel wall-clock이 branch latency 합에 근접하지 않음
- [ ] 한 branch 실패가 다른 성공 결과를 폐기하지 않음
- [ ] cancellation 후 pending task 0개
- [ ] branch status와 total trace가 내부적으로 일관됨
- [ ] deadline measurement boundary가 benchmark 문서와 일치
- [ ] circuit/admission/kill-switch test 통과
- [ ] `runtime_semantics_version=v1` freeze

## Freeze Rule

이후 scheduler 의미, latency timing boundary, parallel/serial 정책을 변경하면 profiler raw data와 latency model을 모두 폐기하고 재생성한다.

---

# Stage 8 — Query Analyzer & Rule Planner

## 목표

가벼운 deterministic analysis로 query feature와 seed entity를 한 번 생성하고, model 없이도 설명 가능한 adaptive plan을 선택한다.

## Entry Criteria

- Stage 1 QueryAnalysis/plan contract 완료
- Stage 7 deadline semantics 동결
- Stage 4와 동일 entity normalization pipeline 사용 가능

## 구현 작업

### S8-001 QueryAnalysis

한 번의 analysis에서 다음을 생성한다.

```text
normalized query
language_supported
token count
query embedding
seed entity mentions/IDs
numeric feature vector
analyzer version
timings
```

### S8-002 Feature schema v1

Feature 범위는 `[0, 1]` 또는 명시적 integer로 고정한다.

```text
token_count
entity_count
entity_density
relation_signal
multi_hop_signal
comparison_signal
aggregation_signal
global_signal
final_top_k
```

- keyword/regex 목록은 versioned config
- English-only analyzer
- query embedding은 model feature로 사용할지 여부를 config에 명시하되 trace에는 미저장

### S8-003 Rule planner

- validation에서만 threshold tuning
- query feature와 remaining budget 사용
- static per-plan safe latency profile 사용 가능
- low budget/unsupported language는 P0 선호
- multi-hop은 budget에 따라 P6/P8
- relation은 budget에 따라 P4/P5/P6
- graph extraction audit gate 실패 또는 graph degraded이면 vector-only plan으로 제한
- stable tie-break

### S8-004 Planner explanation

다음을 `PlannerDecision`에 기록한다.

```text
selected plan
matched rules
remaining budget
candidate feasibility
fallback reason
feature/config version
```

## 주요 파일

```text
src/ragplan/planner/analyzer.py
src/ragplan/planner/features.py
src/ragplan/planner/rule.py
configs/default.yaml
tests/unit/planner/test_analyzer.py
tests/unit/planner/test_rule.py
```

## 테스트

- simple definition query
- relationship query
- comparison query
- 2-hop/3-hop query
- entity-less query
- unsupported language
- max length query
- same query + different budget changes plan where feasible
- exact feature range/schema snapshot
- embedding/entity extraction executed once
- rule explanation deterministic

## Exit Criteria

- [x] `feature_schema_version=qf_v1` freeze
- [x] representative fixture의 feature와 selected plan이 deterministic
- [x] unsupported language가 safe vector plan 사용
- [x] graph audit/dependency disable 상태에서 graph plan을 선택하지 않음
- [x] model 없이 rule adaptive end-to-end 동작
- [x] 모든 decision에 선택 이유가 존재

---

# Stage 9 — Benchmark Harness V1

## 목표

동일 engine/runtime/corpus에서 모든 baseline을 비교하고, profiler와 final report가 재사용할 raw data schema를 완성한다.

## Entry Criteria

- Stage 2 frozen dataset/qrels
- Stage 7 runtime semantics freeze
- Stage 8 rule planner 완료
- Vector/Graph/Fixed mode integration test 통과

## 구현 작업

### S9-001 Benchmark config

다음을 모두 config에 기록한다.

```text
dataset/split/qrels version
corpus version
embedding/extractor version
plan catalog hash
planner config hash
runtime semantics version
hardware/runtime manifest
warmup/repetition/concurrency
random seed
```

- primary run은 CPU-only, 단일 host, local Docker network, concurrency 1
- competing workload 금지와 CPU governor/container resource limit 기록
- query/method 순서를 seed 20260809로 block-randomize
- 환경 또는 DB tuning이 바뀌면 새 run_id로 전체 재실행

### S9-002 Runner

- resumable query execution
- per-query failure가 전체 run을 손상하지 않게 raw status 기록
- duplicate run protection
- timeout/error/fallback 구분
- concurrency 1 primary run

### S9-003 Raw record schema

각 measured trial을 별도 row로 저장한다.

```text
run_id, trial_id, query_id, split
method, planner, plan_id
quality metrics
phase latencies, total latency
branch statuses
timeout/fallback/budget violation
all version hashes
```

### S9-004 Aggregation

- Recall@5/10, MRR@10, nDCG@10
- p50/p95/p99
- timeout/fallback/error/budget violation rates
- overall + query type + dataset source
- confidence interval/bootstrap summary
- percentile은 Hyndman–Fan type 7(`linear`) 사용
- query-cluster paired bootstrap 10,000회, seed 20260809
- 한 query의 measured trial을 같은 bootstrap cluster로 취급
- timeout/error/no-result quality는 0, latency와 rate denominator에는 포함
- outlier 제거와 성공 request만의 latency 재집계 금지

### S9-005 Baselines

```text
Vector-only
Graph-only depth 1/2/3
Fixed Hybrid P4/P5/P6/P8
Rule Planner
BestFixed@Budget selected on validation
```

Test split에서는 validation에서 선택된 BestFixed configuration을 변경하지 않는다.

### S9-006 Cold/warm protocol

- cold run 1회 별도
- warmup 2회
- measured 10회
- raw trial을 삭제하지 않음
- DB tuning 전/후 결과를 같은 table에 섞지 않음

## 주요 파일

```text
benchmark/runners/runner.py
benchmark/runners/records.py
benchmark/metrics/
benchmark/analysis/aggregate.py
benchmark/configs/baseline_v1.yaml
scripts/benchmark.py
tests/benchmark/
```

## 테스트

- interrupted run resume
- duplicate query/run detection
- raw row count validation
- aggregation percentile fixture
- timeout/error denominator
- version mismatch refuses aggregation
- validation-selected config cannot be changed in test command
- same raw input produces byte-stable aggregate JSON where ordering applies

## Exit Criteria — Baseline Evidence Gate

- [ ] 480-query train/validation baseline run 완료; held-out test 120 query는 Stage 14까지 미실행
- [ ] 모든 method의 raw/aggregate artifact 생성
- [ ] query type별 quality/latency table 생성
- [ ] BestFixed가 validation에서만 선택됨
- [ ] environment manifest와 raw result checksum 존재
- [ ] vector/graph/fixed/rule 결과를 동일 조건에서 비교 가능

## Competition fallback

Stage 9에서 graph 또는 rule 성능이 예상보다 낮아도 raw 결과를 보존한다. Cost-aware 작업이 중단될 경우 Stage 13을 완료하고 baseline 중심으로 제출한다.

---

# Stage 10 — Offline Plan Profiler

## 목표

각 query에서 각 P0 plan의 실제 quality/latency를 측정해 Oracle@Budget과 model training matrix를 만든다.

## Entry Criteria

- Stage 9 benchmark schema 검증 완료
- `runtime_semantics_version=v1` 동결
- P0 plan catalog hash 동결
- test split은 profiler/tuning에서 제외

## 구현 작업

### S10-001 Matrix generation

- train/validation query × P0-enabled plan P0/P1/P2/P3/P4/P5/P6/P8
- warmup 2 + measured 10 + cold 1
- all trial rows 저장
- query features와 plan features 별도 column

### S10-002 Oracle labels

각 budget에 대해 실제 measured p95가 budget을 만족하는 plan 중 Recall@10 최대 plan을 Oracle로 정의한다.

Tie-break:

```text
higher Recall@10
→ lower p95 latency
→ lower graph depth
→ lower plan ID
```

### S10-003 Profiler integrity

- missing query-plan row 탐지
- version/hash 불일치 탐지
- branch fallback이 발생한 trial 표시
- partial result quality와 full result quality 구분
- invalid trial exclusion reason 저장

## 주요 파일

```text
benchmark/runners/profiler.py
benchmark/analysis/oracle.py
scripts/profile_plans.py
benchmark/results/profile_<run_id>/
tests/benchmark/test_profiler_matrix.py
tests/benchmark/test_oracle.py
```

## 테스트

- small 3-query × 3-plan exact matrix
- missing trial detection
- Oracle feasibility/tie-break fixture
- runtime/plan/corpus hash mismatch rejection
- test split input rejection

## Exit Criteria

- [ ] 모든 train/validation query-plan row 존재
- [ ] trial count가 config와 정확히 일치
- [ ] budget별 Oracle distribution 생성
- [ ] training matrix와 environment manifest checksum 저장
- [ ] profiler가 final engine/scheduler 경로를 사용했다는 trace 증거 존재

---

# Stage 11 — Cost Models & Artifact Lifecycle

## 목표

Query/plan feature로 Recall@10과 conditional p95 execution latency를 예측하고, 잘못된 artifact가 serving에 로드되지 않게 한다.

## Entry Criteria

- Stage 10 complete matrix
- feature schema, plan catalog, corpus/runtime semantics freeze
- test split 접근 없이 train/validation만 사용

## 구현 작업

### S11-001 Training dataset builder

- query feature + plan feature join
- same query의 모든 trial은 같은 split 유지
- missing/NaN/inf validation
- categorical encoding deterministic
- raw embedding 사용 여부를 명시; P0 기본은 사용하지 않음

### S11-002 Quality model

- `HistGradientBoostingRegressor`
- target Recall@10, range clip 0~1
- train fit, validation evaluation
- MAE/RMSE/ranking accuracy
- predicted best-plan policy regret 측정

### S11-003 Latency model

- quantile 0.95 model
- target execution latency
- MAE/RMSE는 보조
- p95 coverage와 underprediction rate가 주 지표
- cold/warm model은 섞지 않고 P0 serving은 warm model 사용

### S11-004 Artifact serialization

- `skops.io` `.skops` 형식만 사용
- repository에 고정된 trusted-type allowlist만 허용
- unknown type이 하나라도 있으면 load 거부
- `pickle`, `joblib`, `cloudpickle` artifact는 load/자동 변환하지 않음
- manifest와 artifact checksum
- atomic save/load

### S11-005 Compatibility checker

- feature schema
- plan catalog hash
- corpus/qrels version
- model/extractor version
- runtime semantics
- dependency versions
- hardware warning

### S11-006 Model report

- train/validation metrics
- residual distribution
- query type별 error
- plan별 error
- budget violation simulation
- feature importance는 설명 자료로만 사용

## 주요 파일

```text
src/ragplan/planner/quality_model.py
src/ragplan/planner/latency_model.py
src/ragplan/planner/artifacts.py
benchmark/analysis/model_report.py
scripts/train_cost_models.py
tests/unit/planner/test_artifacts.py
tests/benchmark/test_model_training.py
```

## 테스트

- query group leakage 검사
- deterministic training under fixed seed
- NaN/inf/missing feature rejection
- prediction range
- artifact checksum corruption
- every compatibility mismatch branch
- unknown/untrusted artifact refusal
- quantile coverage fixture

## Exit Criteria — Model Gate

- [ ] quality/latency model validation report 생성
- [ ] quality MAE <= 0.10, plan-pair ranking accuracy >= 0.70, policy regret <= 0.05
- [ ] latency overall coverage >= 0.90, plan별 coverage >= 0.85
- [ ] severe underprediction rate <= 0.02
- [ ] constant per-plan p95 대비 pinball loss >= 10% 개선
- [ ] simulated policy가 Rule 대비 Recall 저하 <= 0.01, violation 증가 <= 0.02
- [ ] artifact manifest에 모든 required hash 존재
- [ ] corrupted/incompatible artifact load 거부
- [ ] serving에서 model 미존재 시 rule fallback 가능

## Stop Condition

Model이 validation에서 BestFixed/Rule보다 policy regret 또는 budget violation을 개선하지 못하면 cost-aware를 기본값으로 활성화하지 않는다. Stage 12는 연구/비교 mode로만 구현할 수 있다.

---

# Stage 12 — Cost-aware Optimizer

## 목표

모든 candidate plan을 예측하고 remaining deadline 안에서 가장 높은 predicted Recall@10 plan을 선택한다.

## Entry Criteria

- Stage 11 compatible model artifact
- Stage 7 runtime semantics 유지
- Stage 8 QueryAnalysis 재사용

## 구현 작업

### S12-001 Candidate scoring

- enabled P0 plan 모두 quality/p95 latency 예측
- prediction NaN/inf/out-of-range 처리
- candidate별 model version과 inputs hash 기록

### S12-002 Feasibility filter

```text
predicted_p95 + finalization_reserve <= remaining_budget
```

- infeasible reason 기록
- unsupported top-k 거부
- analyzer/planner elapsed가 큰 요청 처리

### S12-003 Plan selection

- max predicted Recall@10
- stable tie-break
- no feasible plan이면 P0 best-effort
- `budget_feasible=false` 기록

### S12-004 Explainability

`PlannerDecision`에 다음을 포함한다.

```text
all candidate predictions
feasible/infeasible
selected plan
selection reason
fallback/model compatibility status
remaining budget
```

### S12-005 Shadow evaluation

- rule decision과 cost-aware decision을 동시에 계산하되 하나만 실행하는 offline/shadow mode
- planner overhead 비교
- decision disagreement matrix

### S12-006 Runtime model guard

- cost-aware execution rolling window 100, minimum sample 20
- budget violation > 0.10 또는 p95 underprediction rate > 0.20이면 artifact 자동 disable
- process 수명 동안 disabled state 유지하고 rule planner fallback
- disable reason, window count, observed rates를 trace/metric에 기록
- operator가 새 compatible artifact를 명시적으로 load하기 전 자동 재활성화 금지

## 주요 파일

```text
src/ragplan/planner/optimizer.py
src/ragplan/core/engine.py
tests/unit/planner/test_optimizer.py
tests/integration/test_cost_aware_search.py
tests/benchmark/test_policy_regret.py
tests/unit/planner/test_runtime_guard.py
```

## 테스트

- all candidates scored
- infeasible plans excluded
- highest quality feasible selected
- deterministic tie-break
- no feasible plan
- model missing/incompatible
- prediction NaN/inf
- tiny remaining budget
- planner overhead measured
- selected decision explanation complete
- rolling guard threshold/disable/fallback

## Exit Criteria

- [ ] cost-aware end-to-end search 동작
- [ ] decision trace에 모든 candidate와 이유 존재
- [ ] incompatible model이 silent 사용되지 않음
- [ ] validation에서 rule/BestFixed/Oracle 비교 report 생성
- [ ] planner overhead가 별도 metric으로 측정됨
- [ ] runtime guard가 bad calibration artifact를 rule로 격리

---

# Stage 13 — API, CLI, Observability & Packaging

## 목표

동일 engine을 안정적인 REST/CLI로 노출하고 clean-machine 사용자가 설치·ingest·검색·benchmark할 수 있게 한다.

## Entry Criteria

- Stage 7 runtime/failure semantics 완료
- Stage 8 rule planner 완료
- Stage 12 완료 여부와 관계없이 baseline engine 안정

## 구현 작업

### S13-001 REST API

```text
POST /v1/search
GET  /health
GET  /ready
GET  /metrics
```

- Pydantic request/response schema
- planner enum
- complete/partial response
- stable error body
- `api_schema_version=v1`, `trace_schema_version=v1`
- request ID propagation
- OpenAPI examples
- query 1~4096 code point, body 32 KiB, top_k 1~50, budget 25~5000 validation
- unknown field 거부

### S13-002 CLI

```text
ragplan search
ragplan ingest
ragplan benchmark
ragplan profile-plans
ragplan train-models
ragplan verify
```

- CLI는 engine/script API를 재사용
- JSON output option
- non-zero exit code policy

### S13-003 Health/readiness

- `/health`: process liveness
- `/ready`: active corpus와 Qdrant 준비 시 HTTP 200
- Neo4j 장애는 body에 `degraded`로 표시하고 rule planner를 vector-only로 제한
- degraded 상태의 명시적 graph request는 HTTP 503
- compatible cost model 부재는 readiness 실패가 아니라 `rule` fallback 상태

### S13-004 Trace logging

- redacted default
- benchmark opt-in
- query hash/length
- embedding 미직렬화
- bounded async/buffered JSONL writer
- log failure가 retrieval을 실패시키지 않음
- rotation/retention config
- 기본 rotation 10 MiB × 5 files
- service mode raw-query logging 설정 자체를 거부

### S13-005 Metrics

P0 JSON metrics:

```text
request count
complete/partial/error count
planner distribution
branch latency histograms
total latency histogram
timeout/fallback/budget violation
model fallback
```

Raw query/entity를 metric label로 사용하지 않는다.

### S13-006 Docker/API startup

- dependency readiness wait
- one top-level driver/client per process
- graceful shutdown/close
- startup model/corpus compatibility validation
- default planner rule
- `RAGPLAN_DISABLE_COST_AWARE`와 `RAGPLAN_FORCE_VECTOR_ONLY` kill switch

### S13-007 Documentation/examples

- README quick start
- architecture diagram
- sample corpus
- vector/graph/fixed/rule/cost-aware example
- model profiling prerequisite 설명
- limitations/non-production scope

## 주요 파일

```text
src/ragplan/api/server.py
src/ragplan/api/routes.py
src/ragplan/api/schemas.py
src/ragplan/api/errors.py
src/ragplan/cli/app.py
src/ragplan/observability/
README.md
examples/
tests/contract/test_openapi.py
tests/e2e/test_compose_search.py
```

## 테스트

- valid/invalid API request
- partial 200
- empty result 200
- deadline 504
- dependency 503
- internal error redaction
- API/CLI result parity
- raw query/embedding not logged
- trace write failure isolation
- readiness state matrix
- graceful shutdown with in-flight request

## Exit Criteria

- [ ] OpenAPI schema snapshot 승인
- [ ] CLI/API가 동일 query/config에서 동일 plan/result ID 반환
- [ ] default fresh install은 rule planner로 작동
- [ ] partial/error semantics가 Addendum과 일치
- [ ] README 명령만으로 sample ingest/search 성공
- [ ] Docker Compose E2E 통과

---

# Stage 14 — Final Evidence, Reproduction & Submission Freeze

## 목표

기능 개수보다 재현 가능한 증거를 완성하고, 성공 gate를 통과했는지 정직하게 판정한다.

## Entry Criteria

- Stage 9 baseline evidence
- Stage 12 cost-aware optimizer 또는 명시적인 실패/제외 결정
- Stage 13 public surface와 clean Compose path
- test split 아직 tuning에 사용되지 않음

## 구현 작업

### S14-001 Test split final run

- frozen code/config/model/corpus
- test split 단 한 번의 선택된 final run
- raw trials, logs, environment manifest 보존
- 실패 시 bug fix는 가능하지만 tuning 변경은 새 experiment version으로 분리

### S14-002 Required comparisons

```text
Vector-only
Graph-only
Best Fixed Hybrid
Rule Planner
Cost-aware Planner
Oracle@Budget
```

Budget 50/100/200/500 ms별 결과를 생성한다.

### S14-003 Required analysis

- main quality/latency table
- query type table
- budget별 plan distribution
- Pareto chart
- expected vs actual latency calibration
- timeout/fallback/budget violation
- policy regret
- cold/warm separation

### S14-004 Ablation

최소:

```text
without query features / BestFixed
without latency model / Rule
without dynamic depth
without parallel execution
without fallback
```

P7 reranker가 P0에 없으므로 reranker ablation을 필수 결과에 넣지 않는다.

### S14-005 Success gate evaluation

Addendum의 functional, budget, Pareto, query-type guardrail을 자동 판정하는 machine-readable report를 만든다.

```text
PASS
FAIL
NOT_APPLICABLE
```

각 gate에 raw evidence path와 checksum을 연결한다.

### S14-006 Clean-machine reproduction

빈 checkout에서 다음을 검증한다.

```text
dependency install
docker compose up
sample ingest
all baseline search modes
small benchmark smoke
artifact compatibility
service shutdown
```

### S14-007 OSS/license audit

- source header/NOTICE 요구 확인
- dependency/model/dataset attribution
- forbidden/untracked binary
- secret scan
- license/security scan 결과 보존

### S14-008 Submission assets

- README first screen
- architecture diagram
- development report
- raw result manifest
- benchmark charts
- 3-minute demo script/video
- known limitations

## 주요 파일

```text
benchmark/results/final_<run_id>/
benchmark/analysis/final_report.py
benchmark/analysis/success_gate.py
docs/architecture.md
docs/benchmark.md
docs/reproduction.md
docs/limitations.md
scripts/verify_reproduction.py
```

## 검증 명령 목표

```bash
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest tests/unit tests/contract
uv run pytest -m integration
uv run pytest -m e2e
uv run ragplan verify
```

## Exit Criteria

- [ ] 모든 기능/품질 gate가 PASS 또는 명시적 FAIL로 기록
- [ ] test set tuning 없음
- [ ] raw result와 chart가 checksum으로 연결됨
- [ ] clean-machine reproduction 성공
- [ ] license/secret scan 완료
- [ ] README와 report가 측정하지 않은 수치를 주장하지 않음
- [ ] 3분 이내 demo가 네 핵심 장면을 포함

## Freeze Rule

제출 48시간 전부터 다음 변경을 금지한다.

```text
dataset/qrels
chunking/embedding
graph schema/extractor
plan catalog/fusion
feature schema/model target
deadline semantics
new dependency/framework/backend
```

허용 사항은 bug fix, test, benchmark rerun, 문서, 영상, license/reproduction 수정이다.

---

# Stage 15 — Post-MVP Options

다음은 Stage 14 이후 별도 ADR과 benchmark가 있을 때만 진행한다.

## P1

- P7 cross-encoder reranker
- Korean analyzer/NER/relation extraction 및 Korean benchmark
- local LLM graph extractor adapter
- Prometheus/OpenTelemetry
- fuzzy/entity-linking resolver
- P0 admission bound를 넘는 concurrent-load tuning과 autoscaling 연구

## P2

- LanceDB/Milvus/NetworkX adapter
- LangChain/LlamaIndex integration
- online calibration/retraining
- multi-tenant auth/rate limit
- distributed scheduling
- dashboard

P1/P2 변경은 P0 benchmark와 같은 result table에 섞지 않는다.

## 6. Cross-stage quality gates

모든 Stage PR은 관련 범위에서 다음을 충족해야 한다.

```text
Code implemented
Unit/contract test
Error path test
Trace/metric update
Config/schema validation
No raw secrets/query embedding logging
Docs/example update when public behavior changes
CI pass
```

### Contract change rule

다음 파일/계약 변경은 관련 artifact invalidation 검토가 필요하다.

| 변경 | 무효화 대상 |
|---|---|
| ID/chunking/embedding | corpus, Qdrant, qrels, model, benchmark |
| extractor/schema/traversal | Neo4j corpus, model, benchmark |
| plan catalog/fusion | profiler, model, benchmark |
| feature schema | model, profiler join, benchmark decision |
| deadline/scheduler | latency profile, latency model, all performance result |
| dataset/qrels/split | quality model, all quality result |

## 7. 권장 PR/Issue 단위

PR 하나에 여러 Stage를 합치지 않는다. 권장 순서는 다음과 같다.

```text
S0-001 package bootstrap
S0-002 compose and health
S0-003 CI and OSS files

S1-001 domain models
S1-002 plan catalog validation
S1-003 deadline and error contracts
S1-004 backend protocols

S2-001 dataset license/manifest
S2-002 download/preprocess
S2-003 split/qrels
S2-004 metrics

S3-001 chunk/embed
S3-002 qdrant writer
S3-003 vector backend/engine

S4-001 entity/relation extractor
S4-002 neo4j schema/writer
S4-003 ingestion manifest/reconcile
S4-004 extraction quality audit

S5-001 seed lookup/traversal
S5-002 chunk recovery/scoring
S5-003 graph trace/tests

S6-001 common hit/dedup
S6-002 weighted RRF
S6-003 fixed hybrid

S7-001 state/deadline scheduler
S7-002 cancellation/native timeout
S7-003 fallback/trace
S7-004 circuit breaker/admission/kill switch

S8-001 analyzer/features
S8-002 rule planner/explanation

S9-001 benchmark records/runner
S9-002 aggregation/baselines
S9-003 warm/cold reproducibility

S10-001 plan profiler
S10-002 oracle analysis

S11-001 quality model
S11-002 p95 latency model
S11-003 artifact compatibility

S12-001 optimizer selection
S12-002 shadow/policy evaluation
S12-003 runtime model guard

S13-001 API contracts
S13-002 CLI
S13-003 observability/readiness
S13-004 Docker E2E/docs

S14-001 final benchmark
S14-002 ablation/Pareto/gates
S14-003 clean reproduction/license/submission
```

## 8. 최종 구현 시작 체크리스트

Stage 0 착수 전:

- [ ] Addendum을 원 PRD의 우선 결정문으로 인정
- [ ] 영어-only P0와 local/demo 범위를 인정
- [ ] 600-query composite benchmark source를 인정
- [ ] deterministic graph extractor의 품질 한계를 인정
- [ ] cost-aware model이 per-corpus/runtime artifact임을 인정
- [ ] adaptive P0 top-k가 10으로 제한됨을 인정
- [ ] success gate 미달 시 rule planner로 제출한다는 원칙을 인정

Stage 10 착수 전:

- [ ] `runtime_semantics_version=v1` freeze
- [ ] plan catalog hash freeze
- [ ] corpus/qrels/split freeze
- [ ] feature schema freeze
- [ ] train/validation만 profiler 대상으로 사용

Stage 14 착수 전:

- [ ] test split 미사용 확인
- [ ] final artifact/config 선택 완료
- [ ] success gate script dry-run 완료
- [ ] clean-machine smoke 완료

이 체크리스트 중 하나라도 충족되지 않으면 downstream 성능 수치를 공식 결과로 사용하지 않는다.
