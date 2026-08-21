**한국어** | [English](README_EN.md)

# RAGPlan

벡터 및 그래프 검색을 위한 지연 시간 예산 인식 실행 계획 도구입니다.

RAGPlan은 실험용 로컬 우선 검색 최적화 도구입니다. 벡터, 그래프, 고정 하이브리드,
규칙 기반 또는 비용 인식 검색 계획을 선택·실행하고, 실행 추적과 함께 순위가 매겨진
근거를 반환합니다. 답변 생성, 에이전트 루프, 인증, 멀티테넌시 및 프로덕션 배포는 MVP
범위에 포함되지 않습니다.

## 구현된 기반

Stage 0, 1, 2, 3, 4, 5, 6, 7, 8, 9는 재현 가능한 부트스트랩, 공통 런타임 계약, 동결된
benchmark truth, 벡터·그래프 검색, deterministic fixed-hybrid fusion, 예산 인식 병렬 실행
및 설명 가능한 rule planning을 제공합니다.

- Python 3.12 패키지 및 `ragplan` CLI
- FastAPI liveness 엔드포인트
- 고정된 의존성 lockfile
- Docker Compose를 통한 로컬 Qdrant 및 Neo4j 서비스
- lint, 타입 검사 및 테스트 구성
- 오픈 소스 기여, 보안 및 라이선스 파일
- 동결된 Pydantic 도메인/요청/응답/추적 계약
- 결정론적인 표준 문서, 청크, 엔터티 및 Qdrant ID
- 표준 SHA-256 식별자를 갖는 검증된 P0–P8 계획 카탈로그
- ADR-010 reserve를 갖는 하나의 주입 가능한 단조 절대 deadline
- 안정적인 오류/HTTP 매핑 및 분리된 런타임/수집 백엔드 프로토콜
- 결정론적 Unicode 정규화 및 40-token overlap을 갖는 220-token 청크
- 고정 revision에서 체크섬을 검증하는 로컬 전용 `all-MiniLM-L6-v2` 임베딩
- UUIDv5 포인트 및 정확한 reconciliation을 갖는 corpus-version 분리 Qdrant collection
- 분석, 임베딩, Qdrant, 전체 지연 시간 trace를 제공하는 명시적 벡터 모드 실행
- 재현 가능한 모델 준비, 샘플 수집 및 검색 명령
- `adaptive_rag_bench_v1`의 고정 600-query manifest와 360/120/120 group split
- 운영 220/40 청커로 생성한 graded chunk qrels 및 순수 Recall/MRR/nDCG 함수
- checksum·license·attribution audit와 별도 100-query synthetic graph fixture
- 고정 spaCy NER와 deterministic SVO/passive/copular/appositional 관계 추출
- parameterized Cypher, 멱등적 batch write 및 transaction timeout을 적용한 Neo4j writer
- Qdrant/Neo4j 실측 ID reconciliation 뒤에만 전환되는 원자적 active corpus pointer
- 100개 train 문장 human-review queue와 fail-closed rule graph-tier 정책
- exact normalized seed, 1–3 hop `RELATES_TO` traversal 및 별도 `MENTIONS` chunk recovery
- 5 seed/seed당 50 path/500 entity/100 chunk hard cap과 deadline-bound Neo4j transaction
- 방향을 보존하는 path provenance, Graph Score V0 contribution 및 세부 graph phase trace
- canonical chunk ID deduplication과 source별 rank/score/contribution을 갖는 `weighted_rrf_v1`
- P4/P5/P6/P8 및 기본 P5를 실행하는 동일 vector/graph/fixed-hybrid engine
- `runtime_semantics_version=v1` 상태 추적을 갖는 최대 두 branch 병렬 scheduler
- 한 branch 실패 시 성공 branch 순위를 보존하는 typed partial response와 graceful degradation
- 32-request admission, backend별 5-failure/30-second circuit breaker 및 ingress kill-switch snapshot
- backend-client timeout과 application deadline을 구분하고 client disconnect 시 child task를 정리하는 trace
- 동결된 `qf_v1` feature schema와 versioned regex/keyword config를 사용하는 단일-pass query analyzer
- remaining budget, graph audit/circuit 상태 및 정적 p95 profile로 P0/P1/P4/P5/P6/P8을 선택하는 rule planner
- 동일 production engine의 224,640-row baseline matrix를 재개·검증·집계하는 Stage 9 harness

학습 기반 cost-aware planner는 이후 Stage에서 구현됩니다. Stage 3 corpus는 의도적으로
`vector_staged`이며, Stage 4의 두 저장소 검증이 성공해야만 `active`가 됩니다. Human
extraction audit가 완료될 때까지 rule graph tier는 비활성입니다.

## 요구 사항

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker Engine with Docker Compose v2

## 로컬 설정

```bash
cp .env.example .env
uv sync --frozen
uv run ragplan --version
```

`.env.example`의 값은 격리된 로컬 데모 전용입니다. 공유 환경에서 사용하기 전에
Neo4j 비밀번호를 변경하고, `.env`는 절대로 커밋하지 마세요.

## 로컬 스택 시작

```bash
docker compose config --quiet
docker compose up -d --build
curl --fail http://127.0.0.1:8000/health
```

API는 기본적으로 `127.0.0.1`에 바인딩됩니다. 제공된 Compose 구성에서 Qdrant와
Neo4j도 loopback에만 노출됩니다.

## 품질 검사

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not integration and not e2e"
```

로컬 Qdrant 서비스가 정상인 경우 실제 Qdrant 벡터 통합을 실행할 수 있습니다.

```bash
RAGPLAN_TEST_QDRANT_URL=http://127.0.0.1:6333 \
RAGPLAN_TEST_MODEL_SNAPSHOT="$PWD/models/minilm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0" \
uv run pytest \
  tests/integration/test_qdrant_vector_backend.py \
  tests/integration/test_vector_vertical_slice.py
```

## 구성

비밀이 아닌 기본값은 `configs/default.yaml`에 있습니다. 환경변수는 `RAGPLAN_`
접두사와 이중 밑줄을 사용해 중첩 키를 재정의합니다. 예:

```text
server.port       -> RAGPLAN_SERVER__PORT
vector.url        -> RAGPLAN_VECTOR__URL
graph.password    -> RAGPLAN_GRAPH__PASSWORD
```

비밀번호와 기타 비밀값은 환경변수 또는 외부 secret manager로 제공해야 합니다.
커밋된 YAML 파일에 포함하면 안 됩니다.

## Stage 1 계약

정적 검색 계획은 `configs/plans.yaml`에 정의되어 있습니다. P7은 P1 reranker
계획으로 유지되지만 기본 P0 계획 공간에서는 제외됩니다. 로더는 API application을
생성하기 전에 알 수 없거나 누락된 필드, 중복 ID 및 branch/weight/depth/rerank
invariant 위반을 거부합니다.

요청은 명시적인 planner mode인 `vector`, `graph`, `fixed_hybrid`, `rule`,
`cost_aware`를 사용합니다. `adaptive`는 alias가 아닙니다. Query text는 앞뒤
공백이 제거되고 1–4096 Unicode code point로 제한됩니다. Request body는 32 KiB,
top-k는 1–50, latency budget은 25–5000 ms로 제한됩니다. P0 cost-aware 요청은
top-k 10이 필요합니다.

모든 온라인 Stage는 동일한 단조 `Deadline`을 받습니다. Branch 작업은 절대 deadline에서
`min(20 ms, max(5 ms, budget * 0.05))`를 뺀 시점에 중지되며, 숨겨진 grace interval은
없습니다. 기본 trace는 raw query 또는 query embedding을 저장하지 않고 query SHA-256 및
feature summary만 저장합니다.

## Stage 3 벡터 데모

Qdrant를 시작하고 정확한 allowlist 모델 snapshot을 준비한 뒤, 포함된 세 문서 corpus를
수집합니다.

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

모델 명령은 revision `b8903db39f65d93ae28d49a37c4f3fa90c5f94e0`의 manifest
allowlist만 다운로드한 다음 모든 파일의 SHA-256을 검증합니다. ingest 명령은 모든 문서
청크를 하나의 batch로 임베딩하고, Qdrant schema, count, canonical-ID checksum을
검증한 뒤 stage manifest를 원자적으로 기록합니다. 동일한 corpus version을 반복하는
것은 멱등적인 no-op입니다. canonical ID 집합, embedding artifact 또는 embedding bytes의
변경은 mutation 전에 거부됩니다. `vector_staged` v2 manifest는 corpus를 정확한 모델
artifact SHA-256 및 모든 canonical-ID/vector 쌍의 집계 checksum 모두에 연결합니다.

주입 가능한 FastAPI search endpoint가 사용하는 것과 동일한 `VectorSearchEngine`
경로로 query를 실행합니다.

```bash
uv run python scripts/search.py \
  --query "Who wrote notes containing an algorithm for the Analytical Engine?" \
  --stage-manifest artifacts/sample-stage3-vector.json \
  --model-snapshot "$MODEL_SNAPSHOT" \
  --top-k 3 \
  --latency-budget-ms 5000
```

응답은 순위가 매겨진 청크와 redacted trace를 포함한 JSON입니다. query hash 및
analysis/embedding/vector/total timing은 포함하지만 raw query 또는 query embedding은
포함하지 않습니다.

검증된 Stage를 Docker가 제공하는 API로 노출하려면 `.env`의 선택적 Stage 3 값 네 개를
container path로 모두 설정한 뒤 API를 재시작합니다.

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

네 값이 모두 없으면 기본 application은 `NOT_READY` 상태로 유지됩니다. 부분적이거나
유효하지 않은 명시적 구성은 startup에 실패합니다. 구성된 startup은 서비스를 제공하기
전에 로컬 모델 checksum과 Qdrant schema, count, ID, embedding provenance를 검증합니다.
벡터 전용 staging evidence를 dual-store active pointer로 승격하지 않습니다.

## Stage 4 그래프 수집과 활성화

Stage 3가 기록한 정확한 chunk JSONL을 pinned spaCy pipeline으로 처리하고 Neo4j에 inactive
graph version으로 적재합니다. 비밀번호는 command argument가 아닌 환경변수로만 전달합니다.

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

Graph writer는 같은 corpus version과 동일 content에서는 멱등적인 no-op이며, 입력 graph
checksum이 달라지면 쓰기 전에 거부합니다. Activation은 두 backend를 다시 읽어 chunk count와
canonical-ID checksum을 비교하고 성공 시에만 local active pointer를 원자적으로 교체합니다.
어느 한쪽 실패 시 이전 active version이 유지되며 검증된 과거 run으로 rollback할 수 있습니다.
동일 입력으로 `ingest_graph.py`를 다시 실행하면 failed partial batch를 멱등적으로 재시도하며,
명시적인 rollback/record 또는 inactive store 폐기는 `scripts/manage_ingestion.py`를 사용합니다.

활성화된 corpus를 API의 vector 경로로 제공하려면 Stage 3 환경변수를 모두 비우고 `.env`의
Stage 4 값 다섯 개를 설정합니다. Startup은 active pointer와 immutable run manifest가 지정한
corpus/count/checksum/model revision이 vector stage와 정확히 일치하지 않으면 실패합니다.

실제 Stage 2 train passage에서 SHA-256 순으로 고정한 100문장 audit queue는
`benchmark/audits/graph_extraction_v1/`에 있습니다. 현재 human review와 20문장 이중 검토가
완료되지 않았으므로 `configs/graph_tier_policy.json`은 의도적으로 graph tier를 비활성화합니다.
검토자가 `reviews_v1.jsonl`을 완료하기 전에는 이 상태를 통과로 바꾸면 안 됩니다.
완료 후 `uv run python scripts/evaluate_graph_audit.py`로 metrics와 정책을 재계산합니다.

## Stage 5 bounded graph 검색

명시적 `graph` 비교 모드는 atomically activated corpus에서만 실행됩니다. Query seed는
수집과 동일한 pinned spaCy/normalization pipeline으로 생성되고, 저장소 lookup은 UUID와
exact normalized alias가 모두 일치할 때만 성공합니다. Traversal은 `RELATES_TO`만 양방향으로
발견하되 반환 relation의 원래 방향을 유지합니다. `MENTIONS`는 traversal이 끝난 뒤 active
corpus chunk를 복구할 때만 사용합니다.

CLI에서 graph-only 경로를 실행할 수 있습니다. 비밀번호는 command argument가 아니라
환경변수로만 전달합니다. `top-k <= 20`은 P2 depth 1, 더 큰 값은 P3 depth 3을 선택하며,
요청값이 preset candidate 수를 넘으면 trace에 request-floor override가 기록됩니다.

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

API graph-only profile은 `.env.example`의 Stage 5 환경변수 여섯 개를 함께 설정하고 Stage 3/4
runtime 값을 비워야 합니다. Startup은 active manifest, graph-stage evidence, extractor version,
Neo4j marker의 corpus/count/checksum을 모두 다시 검증합니다. Unsupported-language graph 요청은
안전하게 거부되며 향후 rule planner가 vector-only plan으로 처리합니다. Human audit가 pending인
동안에도 명시적 비교 mode는 사용할 수 있지만 rule planner의 자동 graph tier는 계속 꺼져 있습니다.

실제 Neo4j에 대한 1/2/3-hop, cycle, 방향 및 provenance 검증은 다음과 같이 실행합니다.

```bash
RAGPLAN_TEST_NEO4J_URI=bolt://127.0.0.1:7687 \
RAGPLAN_TEST_NEO4J_USER=neo4j \
RAGPLAN_TEST_NEO4J_PASSWORD=ragplan-demo-change-me \
uv run pytest -q tests/integration/test_graph_retrieval.py
```

## Stage 6 fixed hybrid와 fusion

`BaselineSearchEngine` 하나가 같은 active corpus에서 명시적 `vector`, `graph`,
`fixed_hybrid`를 실행합니다. Fixed hybrid는 P4/P5/P6/P8 중 하나를 사용하며 생략 시 P5를
선택합니다. Vector와 graph 후보는 canonical chunk ID로만 deduplicate하고
`weight / (60 + 1-based rank)`를 source별로 합산합니다. 동점은 canonical ID 오름차순으로
결정하며 final hit은 각 source의 원래 rank/score, weight, RRF contribution, source metadata와
graph path를 보존합니다. 같은 ID의 text, document ID 또는 공통 document metadata가 다르면
응답을 만들지 않고 corpus consistency 오류로 종료합니다.

명시적 비교 CLI는 active pointer와 양쪽 stage manifest 및 실제 두 저장소의
corpus/count/checksum/model·extractor provenance가 모두 일치하는지 먼저 검증합니다.

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

API에서는 `.env.example`의 Stage 6 환경변수 열 개를 모두 설정하고 Stage 3/4/5 profile을
비웁니다. Human graph audit가 pending인 동안에도 이 명시적 comparison mode는 사용할 수
있지만 rule planner의 자동 graph 선택은 계속 비활성입니다. Stage 7 scheduler는 활성
branch를 같은 absolute deadline 아래에서 동시에 실행합니다.

실제 두 저장소를 함께 사용하는 vertical test는 다음과 같습니다.

```bash
RAGPLAN_TEST_QDRANT_URL=http://127.0.0.1:6333 \
RAGPLAN_TEST_NEO4J_URI=bolt://127.0.0.1:7687 \
RAGPLAN_TEST_NEO4J_USER=neo4j \
RAGPLAN_TEST_NEO4J_PASSWORD=ragplan-demo-change-me \
uv run pytest -q tests/integration/test_fixed_hybrid.py
```

## Stage 7 scheduler, deadline 및 graceful degradation

Stage 6 dual-store profile은 이제 모든 명시적 mode를 공통 scheduler로 실행합니다. 요청은
`received → analyzing → planning → executing → fusing → complete|partial` 상태를 기록하고,
각 branch는 시작·종료 시각, 남은 예산, cancel reason, timeout origin 및 circuit state를
남깁니다. 두 branch가 성공하면 Weighted RRF를 적용하고, 하나만 성공하면 그 branch의 원래
순위를 유지한 HTTP 200 `partial` 응답을 반환합니다. 둘 다 deadline에 실패하면 504, 둘 다
backend 오류면 503입니다. 정상적인 zero-hit 성공은 HTTP 200 `complete` empty 결과입니다.

Scheduler는 process당 in-flight 요청을 32개로 제한하고 queue 없이 초과 요청을
`OVERLOADED`로 거부합니다. Backend별 circuit은 연속 transport/client-timeout 5회 뒤 30초
열리고 하나의 half-open probe만 허용합니다. Application deadline cancellation은 circuit
failure로 세지 않으며 요청 내부 retry는 없습니다. Qdrant/Neo4j client timeout 상한은
30초이며 더 짧은 transaction/request cutoff를 포함한 25–5000ms 예산은 scheduler의 같은
absolute deadline에서 파생합니다.

`RAGPLAN_FORCE_VECTOR_ONLY=true`는 ingress에서 graph/cost-aware 경로를 우회하고,
`RAGPLAN_DISABLE_COST_AWARE=true`는 cost model 선택을 차단합니다. 두 값은 요청마다 한 번만
snapshot되어 trace에 기록됩니다. HTTP client disconnect 또는 parent cancellation은 모든
backend child task를 cancel한 뒤 await하여 orphan task를 남기지 않습니다.

## Stage 8 query analyzer 및 rule planner

`planner=rule` 요청은 고정된 `query_features_v1.json`에 따라 normalized query, 언어 지원
여부, token/entity 수, seed entity와 `qf_v1` signal을 한 번 생성하고 query embedding도 한
번만 계산합니다. Raw query와 embedding은 trace에 저장하지 않습니다. Rule planner는 분석
후 남은 branch budget과 `rule_planner_v1.json`의 정적 p95 profile을 사용합니다. 관계형
질의는 예산에 따라 P4/P5/P6, multi-hop 질의는 P6/P8을 선호하며, 낮은 예산과 지원하지 않는
언어는 P0으로 안전하게 축소됩니다. Threshold provenance는 `validation`으로 고정되어 test
split을 tuning에 사용하는 설정을 허용하지 않습니다.

`configs/graph_tier_policy.json`의 human audit gate가 실패했거나 graph circuit이 열려 있으면
graph-enabled 후보는 feasibility 목록에서 명시적으로 제외되고 P0/P1만 실행됩니다. 현재
저장소의 audit 상태는 의도적으로 pending이므로 기본 rule 요청은 vector-only입니다. 모든
결정은 selected plan, matched rules, 분석 후 remaining budget, 후보별 feasibility, fallback
reason, feature/config version과 선택 이유를 trace에 남깁니다.

## Stage 2 benchmark 재현

원천 데이터는 저장소에 포함되지 않습니다. 다음 명령은 고정 URL에서 다운로드를
재개하고 크기와 SHA-256을 검증한 뒤, upstream train pool 전체에서
`SHA256(source_dataset + ":" + source_query_id + ":20260809)`가 작은 순서로 정확히
600개를 선택합니다.

```bash
uv sync --frozen --group benchmark --group graph-extraction
uv run python scripts/prepare_model.py --cache-dir models/minilm

MODEL_SNAPSHOT="models/minilm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"

uv run python scripts/prepare_benchmark.py \
  --download \
  --model-snapshot "$MODEL_SNAPSHOT"

uv run pytest -q tests/benchmark
```

Primary manifest는 NQ 200, Hotpot bridge 200, Hotpot comparison 100, MuSiQue
3-hop 100으로 구성됩니다. 분할은 source quota와 document/entity group을 보존하며,
test ID는 별도 immutable manifest로 고정됩니다. Synthetic graph 100개는 cycle, hub,
disconnected 및 1/2/3-hop 검증 전용이고 primary aggregate에는 포함되지 않습니다.
원천 라이선스와 checksum은 `benchmark/manifests/licenses.yaml`, exact/near-duplicate
정책은 `benchmark/manifests/corpus_policy_v1.json`에 기록됩니다. 핵심 Stage 2 데이터·계약
산출물 15개의 크기와 SHA-256은 마지막에 기록되는
`benchmark/manifests/artifact_set_v1.json`으로 한 세대에 묶입니다.

## Stage 9 baseline benchmark

Stage 9은 frozen train/validation 480개만 동일한 active corpus와 production engine에서
비교합니다. 실행 matrix는 vector, graph depth 1/2/3, fixed P4/P5/P6/P8, rule에 대해
50/100/200/500ms 예산, cold sweep 1회, warmup 2회, measured 10회를 사용합니다. Held-out
test 120개는 Stage 14 전까지 로드하지 않습니다. Timeout, backend error, partial fallback,
zero-result도 삭제하지 않고 raw row 및 latency/rate 분모에 남습니다.

먼저 전용 CPU host와 container 제한을 기록합니다. 이어서 `.env.example`의 Stage 6
변수 전체가 정확한 `adaptive_rag_bench_v1-corpus-v1` active corpus를 가리키는 상태에서
재개 가능한 run을 시작합니다.

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

Runner는 configuration 또는 hardware/DB tuning drift, held-out test 접근, corpus/count/ID
checksum 불일치, 동시 writer와 다른 내용의 run ID 재사용을 fail-closed로 거부합니다.
완료 시 `benchmark/results/<run_id>/`에 append-only JSONL, raw/aggregate CSV, aggregate JSON,
validation-only BestFixed lock, environment/run/protocol manifest와 checksum을 생성합니다.
집계는 type-7 p50/p95/p99와 query-cluster paired bootstrap 10,000회(seed 20260809)를
사용하며 outlier를 제거하지 않습니다. 상세 계약은 `docs/benchmark.md`에 있습니다.

저장소에는 실제 480-query 측정값을 꾸며 넣지 않습니다. 위 run은 frozen corpus가 두
DB에 수집·활성화되고 전용 측정 환경이 준비된 뒤 실행해야 하며, 대용량 raw 결과는
기본적으로 Git에서 제외됩니다.

## Stage 10 offline plan profiler

Stage 10은 train/validation 480개 각각에 대해 P0-enabled P0/P1/P2/P3/P4/P5/P6/P8을
50/100/200/500ms에서 전수 실행합니다. 각 query-plan-budget은 cold 1회, warmup 2회,
measured 10회를 유지하며 held-out test split은 loader와 runner 양쪽에서 거부됩니다. 공개
API의 plan 계약은 확장하지 않고, benchmark 전용 진입점이 production analyzer, absolute
deadline, Stage 7 scheduler, fallback, fusion 경로를 그대로 사용합니다.

Stage 9과 동일한 전용 환경 manifest 및 Stage 6 active corpus 설정을 사용합니다.

```bash
docker compose --profile benchmark run --rm benchmark profile \
  --run-id plans_20260819 \
  --environment-manifest /opt/ragplan/artifacts/benchmark_environment.json \
  --confirm-dedicated

docker compose --profile benchmark run --rm benchmark profile-aggregate \
  --run-id plans_20260819
```

완료 시 `benchmark/results/profile_<run_id>/`에 append-only raw trial, 평탄화된 query/plan
feature CSV와 canonical JSONL training matrix, budget별 Oracle label/distribution, environment와
모든 derived artifact checksum이 생성됩니다. Oracle은 measured p95 execution latency가
budget 이하인 완전한 plan row 중 Recall@10 최대값을 선택하며, 동률은 낮은 p95, 낮은 graph
depth, 낮은 plan ID 순으로 해소합니다. Timeout/error와 trace/hash 불일치는 삭제하지 않고
명시적인 exclusion reason으로 남깁니다.

## 라이선스

RAGPlan은 Apache License 2.0으로 배포됩니다. 제3자 소프트웨어, 모델 및 dataset은
각자의 라이선스를 유지합니다. `THIRD_PARTY_LICENSES.md`를 참조하세요.
