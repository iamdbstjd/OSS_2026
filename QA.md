# RAGPlan 오픈소스 대회 QA 계획서

> 상태: BLOCKED — 자동화 QA PASS, 수동 QA-201·QA-202 완료 전 시연 승인 보류
> 범위: Stage 0~12 및 Stage 13 공개 기능
> 주요 목표: 재현 가능한 3분 대회 시연

## 1. 목적

이 계획서는 대회 시연에서 사용하는 공개 주장이 반복 가능한 동작으로 뒷받침되는지
검증한다. 단위 테스트 개수만으로 출시 여부를 결정하지 않는다. 모든 핵심 시연 주장은
테스트 케이스, 예상 결과, 복구 절차, 보존된 증거를 갖춰야 한다.

QA 범위는 순위화된 검색 근거까지다. RAGPlan은 LLM 답변 생성, 인증 또는 멀티테넌시,
Stage 12의 `research_only` 모델 제공을 지원한다고 주장하지 않는다.

## 2. 품질 우선순위

| 우선순위 | 의미 | 출시 판정 기준 |
|---|---|---|
| P0 | 영상에서 보여줄 핵심 주장 또는 장애 복구 경로 | 모든 케이스가 통과해야 하며 건너뛸 수 없음 |
| P1 | 계약, 개인정보 보호 또는 재현성을 뒷받침하는 검사 | 통과하거나 승인된 이슈가 있어야 함 |
| P2 | 선택적인 시연 완성도 또는 소요 시간 관찰 | 사유를 기록하면 연기할 수 있음 |

## 3. 테스트 환경

| 프로필 | 인프라 | 목적 |
|---|---|---|
| `fast` | Python 3.12만 사용 | 플래너, 증거, 개인정보 보호, LLM 전달 계약 검사 |
| `vector` | Qdrant 및 체크섬으로 고정된 MiniLM 캐시 | 실제 샘플 수집과 의미 기반 검색 |
| `dual-store` | API, Qdrant, Neo4j, 검증된 활성 코퍼스 | Fixed Hybrid와 공개 API 동작 검사 |
| `failure-injection` | `dual-store` 및 Docker 제어 권한 | Neo4j 장애, 서비스 성능 저하, 복구 검사 |

WSL Ubuntu와 `~/projects/OSS_2026` 저장소를 사용한다. `.env`, 자격 증명, 토큰,
개인정보가 포함된 모델 캐시 경로, 원본 벤치마크 데이터는 절대 기록하지 않는다.

## 4. 안전 제어

1. `RAGPLAN_QA_LIVE=1`을 명시하지 않으면 라이브 테스트를 실행하지 않는다.
2. `RAGPLAN_QA_ALLOW_FAILURE_INJECTION=1`도 함께 설정하지 않으면 Neo4j를 중지하지 않는다.
3. 장애 주입 테스트는 `finally` 블록에서 Neo4j를 다시 시작하고 복구 여부를 검증한다.
4. 테스트는 `docker compose down --volumes`를 실행하거나 코퍼스를 삭제하거나 고정된
   벤치마크 데이터를 변경하지 않는다.
5. Stage 12는 `research_only` 상태를 유지하며 공개 `cost_aware` 요청은 반드시 안전하게
   차단되어야 한다.
6. 건너뛴 P0 라이브 케이스는 촬영용 QA 통과로 인정하지 않는다.

## 5. QA 매트릭스

| ID | 우선순위 | 시나리오 | 예상 결과 | 자동화 위치 |
|---|---:|---|---|---|
| QA-001 | P0 | 50ms `demo-plan` | P0 vector 선택, 검색 미실행, 원문 질의 미포함 | `tests/qa/test_demo_qa.py` |
| QA-002 | P0 | 500ms `demo-plan` | P1 vector 선택, graph 후보 안전 차단 | `tests/qa/test_demo_qa.py` |
| QA-003 | P1 | Stage 9/10 행 수 주장 | 224,640 + 199,680 = 424,320 | `tests/qa/test_demo_qa.py` |
| QA-004 | P0 | Stage 12 배포 주장 | `research_only`, 공개 제공 비활성화, guard 작동 | `tests/qa/test_demo_qa.py` |
| QA-005 | P1 | 공급자 중립 LLM 전달 | 순위화된 근거 유지, 공급자 SDK 미사용 | `tests/qa/test_demo_qa.py` |
| QA-101 | P0 | 공개 Compose 기능 | `/health`, `/ready`, `/metrics`가 v1 계약 충족 | `tests/e2e/test_compose_search.py` |
| QA-102 | P0 | Fixed Hybrid P5 | vector와 graph가 각각 근거를 반환하고 `weighted_rrf_v1` 결과에 실제 기여 | `tests/e2e/test_compose_search.py` |
| QA-103 | P1 | 메트릭 집계 | 검색 후 요청·플래너·지연시간 카운터 증가 | `tests/e2e/test_compose_search.py` |
| QA-104 | P0 | 공개 cost-aware 요청 | HTTP 503 `MODE_UNAVAILABLE` | `tests/e2e/test_compose_search.py` |
| QA-105 | P0 | Vector quickstart | P0 결과에서 Ada Lovelace 근거가 1위 | `tests/e2e/test_compose_search.py` |
| QA-106 | P0 | Neo4j 장애 | `/ready=degraded`, Qdrant 정상, graph 사용 불가 | `tests/e2e/test_compose_search.py` |
| QA-107 | P0 | Neo4j 장애 중 Rule 요청 | vector-only 실행으로 요청 성공 | `tests/e2e/test_compose_search.py` |
| QA-108 | P0 | 장애 중 명시적 graph 요청 | 일관된 HTTP 503 오류 | `tests/e2e/test_compose_search.py` |
| QA-109 | P0 | Neo4j 복구 | `/ready=ready`, graph 모드 복구 | `tests/e2e/test_compose_search.py` |
| QA-201 | P2 | 전체 촬영 리허설 | 모든 장면을 180초 안에 완료 | 수동 스톱워치·영상 검토 |
| QA-202 | P0 | 비밀정보 및 화면 검토 | 비밀번호, 토큰, `.env`, 개인 경로 미노출 | 수동 프레임 검토 |

## 6. 실행 방법

잠금 파일로 고정된 전체 개발 환경을 설치한다.

```bash
uv sync --frozen --python 3.12 --all-groups
```

Docker나 모델 없이 빠른 QA 테스트 모음을 실행한다.

```bash
uv run pytest -q -m "qa and not e2e"
```

표준 품질 게이트를 실행한다.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not integration and not e2e"
docker compose config --quiet
```

미리 설정한 dual-store 시연 환경을 시작하고 비파괴 라이브 QA를 실행한다.

```bash
docker compose up -d --build --wait --wait-timeout 240
RAGPLAN_QA_LIVE=1 \
uv run pytest -q -m "qa and e2e" tests/e2e/test_compose_search.py
```

모델 기반 quickstart와 제어된 장애 주입을 포함하는 전체 시연 QA 명령은 다음과 같다.

```bash
RAGPLAN_QA_LIVE=1 \
RAGPLAN_QA_RUN_QUICKSTART=1 \
RAGPLAN_QA_ALLOW_FAILURE_INJECTION=1 \
uv run pytest -q -m "qa and e2e" tests/e2e/test_compose_search.py
```

전체 테스트 전에 모든 Stage 6 런타임 변수가 검증된 활성 코퍼스를 가리켜야 한다. 런타임
프로필이 없을 때 기본 Compose 설정은 의도적으로 상태가 정상이지만 `not_ready`인 API를
노출한다. 이 상태만으로는 QA-102 또는 QA-106을 수행할 수 없다.

## 7. 시연 전용 검증 사항

- 50ms와 500ms 플래너 예시는 각각 P0와 P1을 선택한다. 사람의 graph 감사가 완료되지 않은
  상태이므로 어느 예시도 graph 자동 라우팅을 시연하는 것은 아니다.
- Fixed Hybrid 장면은 `planner=fixed_hybrid`, `plan_id=P5`를 명시한 요청이다.
- Neo4j 장애가 정상적인 성능 저하로 인정되려면 Qdrant와 활성 코퍼스를 계속 사용할 수
  있어야 한다.
- Stage 12 증거는 모델 제공 성공이 아니라 배포 차단을 보여주는 시연이다.
- `424,320`은 Stage 9 baseline과 Stage 10 profiler 실험 행의 합이며, 단일 벤치마크
  실행에서 나온 수치가 아니다.
- LLM 예시는 메시지를 구성하지만 실제 LLM 공급자를 호출하지 않는다.

## 8. 증거 보존

모든 출시 후보마다 다음 정보를 기록한다.

```text
커밋 SHA
UTC 시각
OS / Python / uv / Docker 버전
QA 명령어와 종료 코드
pytest 결과 요약
장애 주입 전 /ready 응답
장애 발생 중 /ready 응답
복구 후 /ready 응답
민감정보를 제거한 Fixed Hybrid 응답
Stage 12 증거 매니페스트 체크섬
수동 리허설 소요 시간
```

생성된 로그는 `artifacts/qa/<run-id>/` 아래에 저장하며 기본적으로 Git이 무시한다. 검토
후 민감정보를 제거한 작은 요약만 커밋한다. 원문 질의, 비밀정보, 모델 파일, 벤치마크 원본
행을 강제로 추가해서는 안 된다.

## 9. 출시 판정 기준

다음 조건을 모두 만족할 때만 시연을 승인한다.

- 모든 P0 케이스가 건너뜀 없이 통과한다.
- 장애 주입 후 Neo4j가 복구되었음을 확인한다.
- 포맷, 린트, 타입 검사, 비통합 테스트가 통과한다.
- 화면에 표시하는 테스트 개수는 촬영하는 커밋에서 직접 얻은 값이다.
- 작업 트리 상태와 고정된 증거의 식별자를 기록한다.
- 수동 리허설을 180초 안에 완료한다.
- 마지막으로 모든 프레임에서 비밀정보 노출 여부를 검토하고 통과한다.

조건을 하나라도 만족하지 못하면 실행 상태를 `BLOCKED`로 표시하고 실패 증거를 보존한다.
측정하지 않은 주장으로 실패 결과를 대체해서는 안 된다.

## 10. 현재 로컬 실행 기록

날짜: 2026-08-25
기준 커밋: `9b6eb15`

| 검사 | 결과 | 증거 |
|---|---|---|
| Ruff 포맷 | PASS | 192개 파일 포맷 확인 |
| Ruff 린트 | PASS | 모든 검사 통과 |
| 타입 검사 | PASS | 88개 소스 파일에서 문제 없음 |
| 신규 빠른 QA | PASS | 5개 통과, 414개 제외 |
| 비통합 회귀 테스트 | PASS | 400개 통과, 1개 건너뜀, 18개 제외 |
| 실제 백엔드 통합 테스트 | PASS | Qdrant, MiniLM, Neo4j, Hybrid 집중 검사 4개 통과 |
| Docker 연결 및 Compose 상태 | PASS | API, Qdrant, Neo4j 모두 healthy |
| API 준비 상태 | PASS | `ready`, `dual_store_active`, `sample-stage3-v2` 확인 |
| 최종 image 라이브 E2E | PASS | quickstart·장애 주입·복구 포함 5개 통과 |
| Fixed Hybrid 실제 Graph 기여 강화 검사 | PASS | vector 3개, graph 1개, fusion graph 입력 1개 |
| Stage 13 OpenAPI 고정 계약 | PASS | image 내부 SHA-256이 승인값과 일치 |
| 3분 촬영 리허설 | PENDING | QA-201 수동 스톱워치 필요 |
| 영상 비밀정보 프레임 검토 | PENDING | QA-202 최종 영상 필요 |

### QA-BUG-002 — 샘플 코퍼스에 실제 Graph 검색 기여가 없음 (RESOLVED, P0)

기존 `sample-stage3-v1`에서는 Fixed Hybrid P5의 vector와 graph branch가 모두
`succeeded`였지만 graph 근거와 fusion graph 입력이 0개였다. 기존 E2E는 branch status만
확인해 이 빈 성공을 통과시키고 있었다.

원인은 uncased MiniLM tokenizer의 token ID를 decode해 chunk를 만들면서 원문 대소문자를
잃고, spaCy entity/relation 추출 입력이 훼손되는 흐름이었다. 빠른 tokenizer의 offset mapping으로
원문 표기를 보존하도록 chunker를 수정하고, 검증된
`Ada Lovelace collaborated with Charles Babbage.` 관계 문장을 sample corpus에 반영했다.

새 `sample-stage3-v2`는 Qdrant chunk 3개와 Neo4j entity 6개, mention 6개, relation 1개로
staging·reconciliation·activation되었다. 최종 P5 응답은 vector 3개와 graph 1개를 입력받고,
최상위 결과에 vector와 graph source contribution을 함께 보존한다. 강화된 QA-102는 이 조건을
계속 강제한다.

### QA-BUG-003 — GraphPath JSON round-trip 불일치 (RESOLVED, P1)

Graph 결과가 처음 실제 응답에 포함되면서 computed field인 `hop_count`가 JSON에 직렬화되지만
같은 strict 계약 모델이 이를 extra field로 거부하는 문제를 발견했다. 입력 `hop_count`를 숨김
alias로 받아 relation 길이와 일치하는지 검증하고, 출력은 기존 computed field를 유지하도록
수정했다. Stage 13 OpenAPI SHA-256은
`c25657b5ca93d3b2523c0f47ba0cb7a6d7e71fd454acbd12ae2010f7b0db2c53`으로 변하지 않았다.

### 남은 수동 승인 항목

- QA-201: 실제 화면 전환과 내레이션을 포함한 3분 리허설을 아직 측정하지 않았다.
- QA-202: 최종 녹화물이 없으므로 `.env`, 비밀번호, 토큰, 개인 경로 노출 여부를 프레임별로
  검토하지 않았다.

### 해결된 환경 및 도구 이슈

- 이전 QA-BUG-001은 mypy 증분 캐시를 새로 만든 뒤 해소되었으며, 일반 `uv run mypy src`도
  88개 소스 파일에서 통과했다.
- 이전 QA-ENV-001은 WSL 사용자를 `docker` 그룹이 적용된 새 셸로 전환해 해소되었다.
  `docker version`에서 Client와 Docker Desktop Server를 모두 확인했고 라이브 장애·복구
  테스트까지 통과했다.
