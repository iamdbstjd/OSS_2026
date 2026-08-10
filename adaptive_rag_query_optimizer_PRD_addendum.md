# RAGPlan PRD Addendum / Architecture Decision Record

> 문서 상태: Accepted for MVP implementation  
> 기준일: 2026-08-09  
> 대상 문서: [`adaptive_rag_query_optimizer_PRD.md`](./adaptive_rag_query_optimizer_PRD.md)  
> 목적: 원 PRD의 구현 전 미확정 사항을 결정하고, 충돌 시 이 문서가 우선하도록 한다.  
> 구현 단계: [`stage.md`](./stage.md)

## 1. 결정 요약

| ID | 결정 | MVP 기준 |
|---|---|---|
| ADR-001 | 제품 정체성 | RAG 응답 생성기가 아니라 latency-budget-aware retrieval execution optimizer |
| ADR-002 | 프로젝트 이름 | 대외 작업명 `RAGPlan`; Python package와 CLI는 `ragplan` |
| ADR-003 | 배포 범위 | 로컬/단일 테넌트/단일 프로세스 Python 서비스 + 외부 Qdrant/Neo4j |
| ADR-004 | 지원 언어 | P0는 영어만 지원; 한국어는 P1 이후 |
| ADR-005 | Primary benchmark | `adaptive_rag_bench_v1`, Wikipedia 계열 600 query 고정 세트 |
| ADR-006 | Retrieval ground truth | canonical chunk ID 기반 graded qrels와 별도 synthetic path qrels |
| ADR-007 | Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimension, cosine |
| ADR-008 | Graph extraction | spaCy NER + deterministic dependency/SVO relation extractor; online LLM 없음 |
| ADR-009 | ID와 corpus lifecycle | UUIDv5 Qdrant ID, canonical string ID, versioned dual-write + reconcile + activate |
| ADR-010 | Latency budget | engine ingress부터 response DTO 완성까지의 soft end-to-end deadline |
| ADR-011 | Planner mode | `vector`, `graph`, `fixed_hybrid`, `rule`, `cost_aware`를 명시적으로 구분 |
| ADR-012 | Adaptive output size | P0 adaptive planner는 final `top_k=10`만 지원 |
| ADR-013 | Plan catalog | 완전한 immutable plan 정의; request deadline과 예측값은 plan에서 분리 |
| ADR-014 | Graph traversal | `RELATES_TO`만 탐색, depth 1~3, bounded fan-out, deterministic ranking |
| ADR-015 | Fusion | 프로젝트 고유 `weighted_rrf_v1`, 1-based rank, `k=60` |
| ADR-016 | Cost models | Recall@10 quality model + conditional p95 execution-latency model |
| ADR-017 | Model portability | model은 corpus/runtime 종속 artifact; mismatch 시 rule planner로 fallback |
| ADR-018 | Partial failure | 단일 branch 성공은 HTTP 200 partial, 양쪽 deadline 실패는 HTTP 504 |
| ADR-019 | Logging/privacy | raw query와 embedding은 기본 미기록; benchmark에서만 명시적 opt-in |
| ADR-020 | 재현성과 라이선스 | dependency/image/model/data를 lockfile·digest·checksum으로 고정 |
| ADR-021 | 통계와 success gate | paired query-cluster bootstrap CI로 Pareto 개선과 budget 준수를 판정 |
| ADR-022 | Model acceptance | quality ranking, p95 calibration, validation policy 기준을 모두 통과해야 활성화 |
| ADR-023 | 운영 안전장치 | rule/vector kill switch, circuit breaker, no-retry, admission limit |

### Unknown closure matrix

| 원 분석의 미확정 영역 | 확정 위치 | 닫힌 결정 |
|---|---|---|
| 제품 경계·명칭·언어 | ADR-001~004 | retrieval optimizer, `RAGPlan`, 영어 P0 |
| dataset·license·query 수 | ADR-005 | 3개 upstream train source, 600 query, license와 deterministic selection |
| qrels·split·leakage | ADR-006 | chunk-level graded qrels, 360/120/120, grouped split, cross-source overlap 0 |
| chunk·embedding·extractor | ADR-007~008 | 220/40, pinned MiniLM revision, deterministic spaCy graph extraction |
| Qdrant ID·dual-store 일관성 | ADR-009 | UUIDv5 point ID, versioned dual-write/reconcile/activate |
| latency 측정·budget 의미 | ADR-010, ADR-021 | engine E2E boundary, monotonic clock, no hidden tolerance, fixed aggregation/CI |
| planner/API/top-k/plan invariant | ADR-011~013, ADR-018 | 5개 mode, adaptive top-k 10, immutable plan, validation/error contract |
| graph bound·ranking·fusion | ADR-014~015 | traversal caps, deterministic score/tie, custom weighted RRF |
| profiler·model target·artifact | ADR-016~017, ADR-022 | Recall@10 + conditional p95, fixed trials, `.skops`, compatibility/acceptance gate |
| fallback·rollback·availability | ADR-018, ADR-023 | partial/504/503, kill switch, circuit breaker, admission bound |
| privacy·observability | ADR-019, ADR-023 | redacted-by-default, no embedding log, bounded rotation, reason-code metrics |
| release 판정·통계 | ADR-021 | fixed budgets, paired cluster bootstrap, numeric Pareto/non-regression gates |

정확한 Qdrant/Neo4j patch tag와 image digest처럼 실행 호환성 시험으로만 정할 수 있는 값은 Stage 0 산출물에서 확정한다. 이는 열린 제품 결정이 아니라, Stage 0 Exit 전에 반드시 채워야 하는 재현성 artifact다.

## 2. 제품 및 범위 결정

### ADR-001 — 제품 경계

**Decision**

RAGPlan은 다음까지만 책임진다.

```text
query
→ retrieval plan 선택
→ vector/graph evidence 검색
→ fusion/rerank
→ ranked chunks + paths + trace 반환
```

최종 답변 생성, hallucination 검증, agent loop, web search는 MVP 범위가 아니다. 문서와 발표에서는 `Graph reasoning engine`보다 `graph-augmented evidence retrieval`이라는 표현을 사용한다.

**Why**

Retrieval 성능과 latency trade-off를 독립적으로 측정하고, 경쟁 제품과의 경계를 명확히 하기 위해서다.

### ADR-002 — 이름

**Decision**

- 대외 작업명: `RAGPlan`
- 저장소 가칭: `ragplan`
- Python import: `ragplan`
- CLI: `ragplan`

`AdaptiveRAG`는 기존 NAACL 2024 Adaptive-RAG와 혼동되므로 사용하지 않는다. 최종 공개 전 GitHub/PyPI 이름 가용성을 다시 검사하되, 코드 내부 식별자는 `ragplan`으로 고정한다.

### ADR-003 — 배포 모델

**Decision**

- Python 3.12 modular monolith
- REST: FastAPI + Pydantic v2
- CLI: Typer
- Vector store: Qdrant
- Graph store: Neo4j Community 5.26 LTS 계열
- API, CLI, benchmark는 동일한 `RAGPlanEngine.search()` 경로를 사용
- 기본 bind address는 `127.0.0.1`
- 인증, TLS termination, multi-tenancy, distributed execution은 P0 제외

**Consequence**

P0 결과는 contest/local reproducible MVP이며 production-ready service라고 표시하지 않는다.

Production traffic launch, user feedback collection, online learning, canary rollout은 P0 범위가 아니다. Stage 12의 shadow mode는 frozen offline query를 대상으로 한 decision 비교만 의미한다.

### ADR-004 — 지원 언어

**Decision**

P0의 corpus, query analyzer, NER, relation extractor, benchmark는 영어만 지원한다. 영어가 아닌 query는 HTTP 422가 아니라 정상 처리하되 `language_supported=false`를 trace에 기록하고 rule planner의 vector-only safe plan을 사용한다.

한국어 analyzer/extractor와 한국어 benchmark는 P1이다.

## 3. Benchmark 및 데이터 계약

### ADR-005 — Primary benchmark

**Decision**

`adaptive_rag_bench_v1`은 정답과 supporting fact가 공개된 upstream train split에서 추출한 600개 query로 고정한다. Upstream test split과 hidden label은 사용하지 않는다.

| Source pool | Query 수 | 주 용도 | 라이선스 |
|---|---:|---|---|
| DPR-formatted Natural Questions train | 200 | semantic/entity/single-hop | [CC BY-SA 3.0](https://ai.google.com/research/NaturalQuestions/) |
| HotpotQA distractor train v1.1 — bridge | 200 | relationship/2-hop | [CC BY-SA 4.0](https://hotpotqa.github.io/) |
| HotpotQA distractor train v1.1 — comparison | 100 | comparison/aggregation | [CC BY-SA 4.0](https://hotpotqa.github.io/) |
| MuSiQue-Ans train v1.0 | 100 | 3-hop | [CC BY 4.0](https://github.com/StonyBrookNLP/musique) |

모든 문서는 해당 query가 참조하는 supporting/distractor Wikipedia passage만 내려받아 frozen corpus를 만든다. NQ는 DPR record의 모든 positive context와 upstream 순서상 앞의 hard-negative 9개, HotpotQA/MuSiQue는 record에 포함된 모든 context paragraph를 사용한다. 전체 원본 데이터는 저장소에 재배포하지 않고 download/preprocess script와 checksum manifest를 제공한다.

Dataset license audit가 통과하지 못한 source는 코드 구현 전에 중단하고 동등한 공개 dataset으로 교체한다. 교체 후에는 `adaptive_rag_bench_v1`을 새로 생성하고 기존 수치와 혼합하지 않는다.

선택은 source 내부 filtering 후 `SHA256(source_dataset + ":" + source_query_id + ":20260809)` 오름차순으로 수행한다. MuSiQue가 공개한 `dev_test_singlehop_questions_v1.0.json`의 source question ID와 선택된 NQ query ID의 교집합은 0이어야 한다. Source ID가 직접 호환되지 않으면 normalized question exact match와 whitespace/punctuation-normalized hash를 함께 검사한다.

### ADR-006 — Split과 qrels

**Decision**

- Split: train 360 / validation 120 / test 120
- Split seed: `20260809`
- Source별 train/validation/test 수는 NQ `120/40/40`, Hotpot bridge `120/40/40`, Hotpot comparison `60/20/20`, MuSiQue `60/20/20`
- 동일 query의 plan row는 반드시 같은 split에 속함
- document/entity cluster를 기준으로 group split하여 직접적인 corpus leakage를 방지
- test split은 최종 benchmark 전까지 threshold/model 선택에 사용하지 않음

Query taxonomy:

```text
semantic
entity
relationship
comparison
2hop
3hop
```

Taxonomy는 상호 배타 class가 아니라 multi-label slice다. Manifest에 각 tag의 판정 근거를 저장하며 test split에서 각 필수 tag는 최소 15 query를 포함해야 한다. Group leakage 제약 때문에 quota를 못 맞추면 선택 pool을 확대하고 다시 생성하며, group을 쪼개서 quota를 맞추지 않는다.

Tag assignment는 다음처럼 고정한다.

| Tag | Deterministic rule |
|---|---|
| `semantic` | NQ이며 pinned NER가 허용 entity를 0개 찾음 |
| `entity` | NQ이며 pinned NER가 허용 entity를 1개 이상 찾음 |
| `relationship` | HotpotQA `type=bridge` |
| `comparison` | HotpotQA `type=comparison` |
| `2hop` | HotpotQA이며 supporting title이 2개 이상 |
| `3hop` | MuSiQue decomposition step이 정확히 3개 |

Qrels schema:

```text
query_id
canonical_chunk_id
relevance_grade       # 0, 1, 2
query_tags            # string array
source_dataset
supporting_fact_id
```

- grade 2: 정답 supporting sentence가 들어 있는 chunk
- grade 1: supporting paragraph와 직접 겹치는 chunk
- Recall/MRR의 relevant 기준: grade >= 1
- nDCG는 0/1/2 grade 사용
- chunking 설정이 바뀌면 qrels와 corpus version을 함께 재생성

Primary qrels는 upstream supporting-fact annotation을 deterministic하게 chunk에 투영하므로 새 relevance human annotation을 만들지 않는다. Sentence가 여러 chunk에 overlap되면 해당 sentence를 완전히 포함하는 모든 chunk를 grade 2로 지정한다. Supporting paragraph에 속하지만 supporting sentence를 완전히 포함하지 않는 chunk는 grade 1이다. Mapping이 불가능하거나 source annotation이 모순인 query는 임의 보정하지 않고 selection pool의 다음 query로 교체하며 exclusion reason을 남긴다.

별도 synthetic graph benchmark 100 query를 제공하며 1/2/3-hop path qrels를 포함한다. Synthetic 결과는 primary success claim에 합산하지 않는다.

## 4. Ingestion 및 저장 계약

### ADR-007 — Embedding과 chunking

**Decision**

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Revision: `b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`
- Dimension: 384
- Distance: cosine
- 위 revision의 downloaded artifact별 SHA256을 manifest에 기록하고 floating `main`을 load하지 않음
- Chunk size: tokenizer 기준 220 tokens
- Overlap: 40 tokens
- 빈 chunk와 whitespace-only chunk는 저장하지 않음

Query embedding은 `QueryAnalysis`에서 한 번만 계산하고 vector retrieval에서 재사용한다. Embedding vector는 trace와 JSONL에 직렬화하지 않는다.

### ADR-008 — Entity/relation extraction

**Decision**

P0 기본 extractor는 외부 API나 online LLM을 사용하지 않는다.

- Entity: pinned `en_core_web_sm` NER
- 허용 type: `PERSON`, `ORG`, `GPE`, `LOC`, `FAC`, `PRODUCT`, `EVENT`, `WORK_OF_ART`
- Relation: dependency parse 기반 deterministic SVO/passive/copular/appositional rules
- Edge: `(:Entity)-[:RELATES_TO]->(:Entity)`
- Edge properties: `predicate`, `confidence`, `source_chunk_id`, `extractor_version`
- Relation minimum confidence: `0.70`

`spaCy`, `en_core_web_sm`, tokenizer wheel의 정확한 버전은 Stage 0 lockfile에 기록하고 extractor version은 해당 버전, normalization rule, relation rule source hash의 조합으로 계산한다. Lockfile이 없는 extractor 실행은 benchmark에서 거부한다.

Entity normalization:

```text
Unicode NFKC
→ trim
→ whitespace collapse
→ Unicode casefold
→ surrounding punctuation removal
```

P0는 exact normalized alias match만 사용한다. Fuzzy/entity-linking model은 P1이다.

Graph extraction audit은 train split passage에서 hash로 뽑은 100 sentence를 사용한다. 전부 한 명이 entity span/type과 directed relation을 검토하고, 그중 20개는 두 번째 reviewer가 독립 검토해 agreement와 adjudication을 기록한다. Entity micro-F1이 `0.75` 미만이거나 relation precision이 `0.70` 미만이면 graph mode는 비교 baseline으로만 남기고 rule planner의 graph 선택을 비활성화한다.

### ADR-009 — ID, dual-write, corpus version

**Decision**

Canonical IDs:

```text
document_id = source_dataset + source_document_id
canonical_chunk_id = document_id + chunk_index + content_hash_prefix
entity_id = UUIDv5(entity_type + normalized_name)
qdrant_point_id = UUIDv5(canonical_chunk_id)
```

Qdrant에는 UUID point ID를 사용하고 `canonical_chunk_id`는 payload에 저장한다. Neo4j Chunk node에는 canonical string ID를 저장한다.

Qdrant와 Neo4j 사이 분산 transaction은 구현하지 않는다. 대신 versioned activation을 사용한다.

```text
1. 새 corpus_version을 inactive 상태로 생성
2. Qdrant write
3. Neo4j write
4. document/chunk count와 ID checksum reconcile
5. 성공 시 active corpus_version 전환
6. 실패 시 이전 active version 유지
```

In-place corpus mutation은 금지한다. update/delete는 새 corpus version 생성으로 처리하고, 이전 version 삭제는 별도 maintenance 명령으로 수행한다.

## 5. Online request 및 planner 계약

### ADR-010 — Deadline 의미

**Decision**

`latency_budget_ms`는 다음 구간의 soft end-to-end server processing deadline이다.

```text
RAGPlanEngine.search() 진입
→ query analysis
→ planning
→ retrieval
→ fusion/rerank
→ response DTO 완성
```

클라이언트 네트워크 왕복과 HTTP body 전송 시간은 제외한다.

- default budget: 200 ms
- accepted range: 25~5000 ms
- benchmark budgets: 50, 100, 200, 500 ms
- 모든 단계는 동일한 monotonic absolute deadline을 공유
- clock: Python `time.perf_counter_ns()` 또는 같은 monotonic nanosecond clock
- finalization reserve: `min(20ms, max(5ms, budget * 0.05))`
- branch는 `deadline - finalization_reserve`까지만 실행
- budget은 hard real-time 보장이 아니며 violation rate로 품질을 판정
- 숨은 grace/tolerance는 두지 않으며 `total_latency_ms > latency_budget_ms`이면 violation

Analyzer/planner가 이미 소비한 시간은 optimizer 실행 시 remaining budget에서 차감한다.

### ADR-011 — Planner modes

**Decision**

API와 CLI에서 다음 enum을 사용한다.

```text
vector
graph
fixed_hybrid
rule
cost_aware
```

`adaptive`라는 모호한 alias는 사용하지 않는다.

- clean install 기본값: `rule`
- `cost_aware`: 호환되는 model artifact가 로드됐을 때만 사용 가능
- model이 없거나 호환되지 않으면 명시적 `planner_fallback_reason`과 함께 `rule`로 fallback

### ADR-012 — Final top-k

**Decision**

- API 범위: `1 <= top_k <= 50`
- P0 `cost_aware` planner는 `top_k=10`만 지원
- `cost_aware` + `top_k != 10`은 HTTP 422
- 다른 planner mode는 1~50을 지원하되 competition primary metric은 top-k 10
- plan의 `vector_top_k`/`graph_top_k`는 candidate size이고 request `top_k`는 최종 결과 수

### ADR-013 — Domain contracts와 plan catalog

**Decision**

다음 객체를 분리한다.

```text
SearchRequest
RequestContext / Deadline
QueryAnalysis
PlanDefinition
PlannerDecision
BranchResult
SearchResponse
SearchTrace
IngestionManifest
ModelManifest
```

`PlanDefinition`은 immutable하고 static plan parameter만 가진다. `timeout_ms`, `expected_quality`, `expected_latency_ms`는 plan에서 제거하고 각각 deadline과 `PlannerDecision`에 둔다.

P0/P1 plan catalog:

| ID | Vector k | Graph k | Depth | V/G weight | Rerank | P0 |
|---|---:|---:|---:|---:|---|---|
| P0 VECTOR_FAST | 10 | 0 | 0 | 1.00 / 0.00 | off | on |
| P1 VECTOR_WIDE | 30 | 0 | 0 | 1.00 / 0.00 | off | on |
| P2 GRAPH_SHALLOW | 0 | 20 | 1 | 0.00 / 1.00 | off | on |
| P3 GRAPH_DEEP | 0 | 30 | 3 | 0.00 / 1.00 | off | on |
| P4 HYBRID_VECTOR_HEAVY | 20 | 15 | 1 | 0.70 / 0.30 | off | on |
| P5 HYBRID_BALANCED | 20 | 20 | 1 | 0.50 / 0.50 | off | on |
| P6 HYBRID_GRAPH_HEAVY | 15 | 30 | 2 | 0.30 / 0.70 | off | on |
| P7 HYBRID_RERANK | 30 | 30 | 2 | 0.50 / 0.50 | on, final 10 | off |
| P8 HYBRID_GRAPH_DEEP | 15 | 40 | 3 | 0.25 / 0.75 | off | on |

Plan invariants:

- 최소 한 branch 활성
- 비활성 branch의 top-k와 weight는 0
- 활성 branch top-k는 1 이상
- weight는 0 이상이고 합은 1
- graph depth는 graph off일 때 0, graph on일 때 1~3
- rerank는 candidate가 있을 때만 활성
- stable tie-break는 lower predicted latency, lower graph depth, lower plan ID 순

P7 reranker는 P1이며 P0 profiler/config에 포함하지 않는다.

### ADR-014 — Graph traversal

**Decision**

- Traversal relationship allow-list: `RELATES_TO` only
- 탐색은 양방향이지만 반환 path에는 원래 edge direction을 보존
- depth: 1~3
- max seed entities: 5
- max paths per seed: 50
- max visited entities: 500
- max recovered chunk candidates: 100
- 동일 path에서 node 반복 금지
- seed lookup은 exact normalized alias만 허용
- `MENTIONS`와 `HAS_CHUNK`는 traversal이 아니라 chunk recovery 단계에서만 사용

Graph ranking V0:

```text
0.45 * normalized_seed_overlap
+ 0.35 * (1 / hop_count)
+ 0.20 * mean_relation_confidence
```

점수가 같으면 `canonical_chunk_id` 오름차순으로 정렬한다.

### ADR-015 — Fusion

**Decision**

프로젝트 fusion 이름은 `weighted_rrf_v1`이다.

```text
score(d) = Σ source_weight / (60 + one_based_rank)
```

- Qdrant server-side weighted RRF와 동일하다고 주장하지 않음
- 결과 정렬: fused score 내림차순, canonical chunk ID 오름차순
- final hit은 `sources`, source별 rank/contribution, graph path를 보존
- 한 branch만 성공하면 해당 branch ranking을 그대로 유지하고 `partial=true` 기록

## 6. Cost model 및 optimizer 계약

### ADR-016 — Model targets와 profiler

**Decision**

Quality model:

- target: `Recall@10`
- model: scikit-learn `HistGradientBoostingRegressor`

Latency model:

- target: scheduler 시작부터 fused result DTO 생성까지의 conditional p95 execution latency
- model: quantile `HistGradientBoostingRegressor`, quantile 0.95
- analyzer/planner 실제 경과시간은 deadline에서 이미 차감

Profiler protocol:

- final scheduler/deadline/fallback 구현 후 실행
- query × enabled P0 plan 전수 실행
- 각 query-plan마다 warmup 2회 + measured 10회
- cold run 1회를 별도 저장
- measured trial을 aggregate 전에 모두 보존
- concurrency 1을 primary benchmark로 사용
- hardware, OS, Python, image digest, DB config, corpus/model/config hash 저장

Optimizer feasibility:

```text
predicted_p95_execution_latency
+ finalization_reserve
<= remaining_budget
```

feasible plan 중 predicted Recall@10이 최대인 plan을 선택한다. Feasible plan이 없으면 P0를 best-effort로 실행하고 `budget_feasible=false`를 기록한다.

### ADR-017 — Artifact compatibility

**Decision**

Model artifact는 다음 manifest를 포함한다.

```text
artifact_version
feature_schema_version
plan_catalog_hash
corpus_version
qrels_version
embedding_model_revision
extractor_version
Qdrant/Neo4j/client versions
training_config_hash
train/validation split hash
runtime fingerprint
validation metrics
```

Quality/latency model은 `skops.io`의 `.skops` 형식으로만 저장한다. Loader는 artifact와 manifest의 SHA256을 먼저 검증하고, repository에 고정된 trusted-type allowlist 밖의 type이 하나라도 있으면 load를 거부한다. `pickle`, `joblib`, `cloudpickle` artifact는 자동 변환하거나 load하지 않는다.

Critical field가 다르면 artifact load를 거부하고 rule planner로 fallback한다. Hardware fingerprint만 다르면 warning을 내고 cost-aware를 기본값으로 활성화하지 않는다.

Quality model은 corpus/query distribution 종속이다. 새 corpus 사용자는 `rule` planner로 시작하고 local profiling/training 후 `cost_aware`를 명시적으로 활성화한다.

## 7. API, failure, observability 계약

### ADR-018 — API와 failure semantics

**Decision**

Request validation:

- `query`: trim 후 1~4096 Unicode code point
- whitespace-only query: HTTP 422 `INVALID_QUERY`
- `top_k`: integer 1~50
- `latency_budget_ms`: integer 25~5000
- unknown planner enum/unknown JSON field: HTTP 422
- request body maximum: 32 KiB

`POST /v1/search` response는 다음 상태를 가진다.

```text
complete
partial
```

Branch status:

```text
not_scheduled
running
succeeded
timed_out
failed
cancelled
```

Public response와 trace는 각각 `api_schema_version="v1"`, `trace_schema_version="v1"`을 포함한다. Error body는 `code`, `message`, `request_id`, `retryable` 네 필드를 항상 포함하고 stack trace나 backend credential을 포함하지 않는다.

HTTP rules:

| 상황 | HTTP | 의미 |
|---|---:|---|
| 모든 활성 branch 성공 | 200 | `status=complete` |
| 한 branch 성공, 다른 branch timeout/error | 200 | `status=partial`, `fallback=true` |
| 정상 검색이지만 zero hit | 200 | 빈 results |
| 전체 deadline 내 결과 없음 | 504 | stable error code `DEADLINE_EXCEEDED` |
| 모든 활성 branch가 backend error | 503 | `DEPENDENCY_UNAVAILABLE` |
| circuit/kill switch로 요청 mode 사용 불가 | 503 | `MODE_UNAVAILABLE` |
| admission 상한 초과 | 503 | `OVERLOADED`, `retryable=true` |
| 필수 dependency/model 준비 안 됨 | 503 | `NOT_READY` 또는 `DEPENDENCY_UNAVAILABLE` |
| 입력 또는 plan invariant 위반 | 422 | validation error |
| 예상하지 못한 내부 오류 | 500 | request ID만 외부 노출 |

`/health`는 process liveness만 나타낸다. `/ready`의 필수 조건과 degraded Neo4j 처리는 ADR-023을 따른다.

### ADR-019 — Trace와 privacy

**Decision**

기본 trace에 저장하는 값:

```text
request_id
query_hash
query_length
language_supported
feature summary
planner mode/decision
candidate predictions
branch status/latency
fusion/rerank latency
total latency
budget feasibility/violation
corpus/config/model versions
```

기본 미저장:

```text
raw query
query embedding
full document text
credentials
exception stack trace in API response
```

Frozen public benchmark에서는 `logging.mode=benchmark`를 명시적으로 설정해 query ID와 raw query를 기록할 수 있다. 일반 실행 기본값은 `redacted`다.

## 8. Reproducibility, license, success gate

### ADR-020 — 버전과 라이선스

**Decision**

- Project license: Apache-2.0
- Python dependency는 lockfile로 고정
- Docker images는 tag와 digest를 모두 기록
- Model revision과 SHA256 기록
- dataset은 download URL, upstream version, license, checksum 기록
- `THIRD_PARTY_LICENSES.md`에 Qdrant, Neo4j, model, dataset, Python dependency를 분리 기록
- 라이선스가 불명확한 artifact는 repository와 Docker image에 포함하지 않음

Qdrant/Neo4j의 정확한 patch tag와 image digest는 Stage 0에서 clean-machine smoke test를 통과한 조합으로 freeze한다. 이후 benchmark 기간에는 변경하지 않는다.

Primary dataset license 기준은 Natural Questions `CC BY-SA 3.0`, HotpotQA `CC BY-SA 4.0`, MuSiQue `CC BY 4.0`이다. 원본 dataset은 repository와 Docker image에 포함하지 않고 사용자가 download script를 실행해 받으며, 파생 manifest/qrels에는 source ID와 attribution을 보존한다.

### ADR-021 — Metric, 통계, quantitative success gate

**Decision**

Aggregation contract:

- quality는 query별로 계산하며 timeout/error/no-result query의 Recall/MRR/nDCG는 0
- partial response의 quality는 실제 반환 결과로 계산
- latency p50/p95/p99는 warm measured trial 전체에서 Hyndman–Fan type 7(`linear`)로 계산
- timeout/error도 engine 진입부터 error DTO 완성까지 latency distribution에 포함
- outlier 삭제, winsorization, 성공 request만의 선택적 집계 금지
- cold 1회는 별도 표에만 보고하고 primary gate에 사용하지 않음
- CI는 query ID를 cluster로 한 paired bootstrap 10,000회, seed `20260809`, percentile 95% CI
- 한 query의 10 measured trial은 bootstrap에서 함께 resample하여 pseudo-replication을 피함

Primary performance run은 CPU-only 단일 reference host에서 API process, Qdrant, Neo4j를 local Docker network에 두고 concurrency 1로 수행한다. 다른 user workload를 동시에 실행하지 않고 query/method 순서는 seed `20260809`의 deterministic block randomization을 사용한다. Hardware, CPU governor, container resource limit 또는 DB tuning이 바뀌면 새 `run_id`로 전부 다시 측정한다.

Budget별 BestFixed는 validation split에서만 고른다. `feasible budget`은 validation에서 적어도 한 fixed plan의 end-to-end p95가 해당 budget 이내인 budget이다. 50/100/200/500 ms를 모두 보고하되, primary 성공 주장에는 feasible budget이 최소 2개이고 그중 하나가 200ms 또는 500ms여야 한다.

Budget violation denominator는 error/timeout을 포함한 전체 measured request다. Network round trip은 제외하지만 engine 내부 serialization과 error DTO 생성은 포함한다.

Functional gate:

- Vector, Graph, Fixed Hybrid, Rule, Cost-aware mode end-to-end 성공
- 600-query frozen test harness 실행 가능
- plan과 branch trace 완전성 100%
- clean-machine Compose reproduction 성공

Budget gate:

- feasible budget의 end-to-end budget violation rate `<= 0.05`
- unexpected request error rate `<= 0.01`
- partial fallback rate `<= 0.05` at 200/500 ms budgets

Pareto gate — held-out test에서 다음 중 하나 이상을 만족:

1. paired 95% CI의 Recall@10 차이 하한이 `+0.02` 이상이고 p95 latency ratio 상한이 `1.05` 이하
2. paired 95% CI의 p95 latency ratio 상한이 `0.85` 이하이고 Recall@10 차이 하한이 `-0.01` 이상

Guardrails:

- 어떤 필수 query tag에서도 BestFixed 대비 Recall@10 point-estimate 감소가 `0.03`을 초과하지 않음
- Oracle@Budget 대비 mean policy regret `<= 0.05` Recall@10
- 수치는 test split 1회 선택 후 재튜닝 없이 산출

Gate를 만족하지 못하면 결과를 숨기거나 threshold를 test set에 맞춰 수정하지 않는다. Competition release의 기본 planner는 `rule`로 유지하고 cost-aware 결과를 제한점으로 보고한다.

### ADR-022 — Cost model acceptance

**Decision**

Cost-aware artifact는 validation split에서 다음을 모두 만족할 때만 serving opt-in 대상이 된다.

Quality model:

- Recall@10 MAE `<= 0.10`
- 동일 query의 plan pair ranking accuracy `>= 0.70`; exact target tie는 denominator에서 제외
- selected-plan mean policy regret vs Oracle `<= 0.05`

Latency model:

- `actual_execution_latency <= predicted_p95` empirical coverage: overall `>= 0.90`, plan별 `>= 0.85`
- actual이 predicted p95의 1.20배를 초과하는 severe underprediction rate `<= 0.02`
- constant per-plan p95 predictor보다 validation pinball loss를 `>= 0.10` 개선

Simulated policy:

- Rule 대비 Recall@10 차이 `>= -0.01`
- Rule 대비 budget violation rate 증가 `<= 0.02`

하나라도 실패하면 artifact status는 `research_only`이며 default나 explicit public API `cost_aware`에 load하지 않는다. Threshold 변경은 test 열람 전에 Addendum version을 올려야 한다.

### ADR-023 — Operational safety와 rollback

**Decision**

- 기본 planner는 항상 `rule`; `RAGPLAN_DISABLE_COST_AWARE=true`이면 model load와 cost-aware 선택을 모두 차단
- `RAGPLAN_FORCE_VECTOR_ONLY=true`는 graph와 cost-aware를 우회하는 최종 kill switch
- request 내부 backend retry는 0회; deadline을 숨겨 소비하는 자동 retry를 금지
- backend별 5회 연속 backend-native transport/timeout 실패 시 circuit을 30초 open, 이후 1개 half-open probe만 허용. 정상적인 request deadline cancellation은 failure count에서 제외
- process당 in-flight request 기본 상한 32, request당 backend task 상한 2
- admission slot을 즉시 얻지 못하면 HTTP 503 `OVERLOADED`; 내부 무제한 queue 금지
- `/health`는 process만 검사. `/ready`는 active corpus와 Qdrant가 준비되면 HTTP 200이며 Neo4j 장애는 `degraded`로 표시하고 rule을 vector-only로 제한. 이때 명시적 `graph`/`fixed_hybrid` request는 503 `MODE_UNAVAILABLE`
- redacted trace는 10 MiB × 5 rotating file이 기본. Raw query는 benchmark artifact에서만 허용하고 일반 service log에는 retention 설정과 무관하게 기록하지 않음
- cost-aware 실제 실행을 최근 100 request rolling window로 감시하되 최소 20개부터 판정. budget violation이 `0.10`을 초과하거나 actual latency가 predicted p95보다 큰 비율이 `0.20`을 초과하면 해당 artifact를 process 수명 동안 자동 disable하고 rule로 fallback

Circuit state 전환, kill switch 발동, overload, model fallback은 trace/metric에 reason code와 함께 기록한다.

## 9. 대안과 결과

### Alternatives considered

- **LLM-based online router:** latency, 비용, 재현성 때문에 P0 제외
- **LLM-based graph extraction:** graph 품질 잠재력은 높지만 artifact/API 의존성 때문에 P1로 연기
- **Qdrant-only hybrid:** 구현은 단순하지만 external graph traversal 실험을 검증할 수 없어 제외
- **Microservice 분리:** 운영 확장성은 높지만 deadline 관리와 MVP 복잡도가 증가하여 제외
- **평균 latency model:** 구현은 쉽지만 p95 budget 제약과 맞지 않아 제외
- **In-place dual-store update:** partial failure 복구가 어려워 versioned activation으로 대체

### Consequences

- Cost-aware model은 설치 직후 범용으로 작동하지 않고 corpus/runtime별 profiling이 필요하다.
- P0 graph extraction 품질은 제한적이지만 완전한 재현성과 분석 가능성을 확보한다.
- 영어 이외 query와 arbitrary adaptive top-k는 제한된다.
- Scheduler와 runtime semantics를 동결하기 전에는 profiler/model 학습을 시작할 수 없다.

### Follow-ups

- P1: reranker P7, 한국어, local LLM extractor, Prometheus, fuzzy entity resolver
- P2: 추가 backend adapter, online calibration, distributed execution, production auth/rate limit

## 10. 변경 통제

이 Addendum의 다음 항목이 바뀌면 corpus/model/benchmark version을 함께 올려야 한다.

```text
chunking
embedding model
entity/relation extractor
graph schema/traversal
plan catalog
fusion formula
feature schema
quality/latency target
dataset/qrels/split
deadline measurement boundary
```

마감 48시간 전부터 위 항목은 변경하지 않는다.
