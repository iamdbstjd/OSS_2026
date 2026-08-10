**한국어** | [English](README_EN.md)

# RAGPlan

벡터 및 그래프 검색을 위한 지연 시간 예산 인식 실행 계획 도구입니다.

RAGPlan은 실험용 로컬 우선 검색 최적화 도구입니다. 벡터, 그래프, 고정 하이브리드,
규칙 기반 또는 비용 인식 검색 계획을 선택·실행하고, 실행 추적과 함께 순위가 매겨진
근거를 반환합니다. 답변 생성, 에이전트 루프, 인증, 멀티테넌시 및 프로덕션 배포는 MVP
범위에 포함되지 않습니다.

## 구현된 기반

Stage 0, 1, 3은 재현 가능한 부트스트랩, 공통 런타임 계약 및 동작하는 벡터 검색
수직 슬라이스를 제공합니다.

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

그래프 수집, 그래프 검색, 하이브리드 융합 및 적응형 planner는 이후 Stage에서
구현됩니다. Stage 3 corpus는 의도적으로 `active`가 아닌 `vector_staged`로
표시됩니다. 활성화에는 Stage 4 dual-store reconciliation contract가 필요합니다.

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
  --stage-manifest artifacts/sample-stage3-vector.json
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

## 라이선스

RAGPlan은 Apache License 2.0으로 배포됩니다. 제3자 소프트웨어, 모델 및 dataset은
각자의 라이선스를 유지합니다. `THIRD_PARTY_LICENSES.md`를 참조하세요.
