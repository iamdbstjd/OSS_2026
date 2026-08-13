# Stage 9 Benchmark Protocol V1

이 문서는 `baseline_v1`의 실행·증거 계약을 설명한다. 실제 측정값이 아니라 측정 방법과
검증 규칙을 고정하며, primary evidence는 반드시 production engine과 활성화된 frozen corpus에서
생성해야 한다.

## 실행 범위

- Dataset: `adaptive_rag_bench_v1`
- Query: train 360 + validation 120 = 480; test 120은 Stage 14까지 접근 금지
- Method: vector, graph depth 1/2/3, fixed P4/P5/P6/P8, rule의 9개 실행 method
- Budget: 50/100/200/500ms
- 반복: cold sweep 1 + warmup 2 + measured 10
- 환경: CPU-only, single host, local Docker network, concurrency 1
- 순서: seed 20260809로 phase/repetition/budget block 안의 query×method를 shuffle

따라서 완전한 raw matrix는 224,640행이다. Cold 17,280행은 별도로 집계하고 warmup
34,560행은 보존하되 발표 집계에서 제외한다. Measured evidence는 172,800행이다.

Graph depth 1과 3은 immutable P2/P3를 사용한다. Catalog에 graph-only depth 2 plan이 없으므로
depth 2는 public API를 확장하지 않는 benchmark-only P3 파생 실행이다. Exact derived plan과 별도
config identity가 trace에 남고, profiler의 static P3 row와 섞지 않는다.

## Evidence identity

`benchmark/configs/baseline_v1.yaml`은 다음 identity를 고정한다.

- Stage 2 artifact-set, dataset manifest, split, qrels logical SHA-256
- corpus version, chunk count, canonical chunk-ID checksum
- embedding revision과 extractor version
- plan catalog, rule config, query feature config, graph-tier policy SHA-256
- runtime semantics, repetition, percentile, bootstrap 규칙

Run manifest는 여기에 protocol SHA-256, environment SHA-256, 정렬된 480 query-ID SHA-256,
생성 시각과 예상 row 수를 결합한다. Raw row는 이 identity bundle 전체를 반복 기록한다.
Aggregation은 단 하나라도 다르거나 trial ID·phase·repetition matrix가 어긋나면 실패한다.

Environment manifest는 OS/CPU/governor, Python, dependency lock, Compose, image refs, container
resource limits, executable source tree와 DB-tuning checksum을 기록한다. 기본 tuning identity는
`benchmark/configs/db_tuning_default_v1.json`이며 설정을 바꾸면 이 versioned file도 바꿔야 한다.
실행 시 현재 환경을 다시 캡처해 동일하지 않으면 새 run ID를 요구한다. Secret과 raw query는
어느 evidence artifact에도 기록하지 않는다.

## Failure and resume semantics

`raw_trials.jsonl`은 canonical compact JSON을 한 행씩 append하고 각 행을 `fsync`한다. Trial ID는
run/query/method/budget/phase/repetition에서 결정론적으로 생성된다. 재시작 시 이미 검증된 trial만
건너뛰며 truncated row, duplicate ID, schedule 밖 row, identity mismatch를 거부한다. 같은 run
directory는 non-blocking exclusive lock으로 writer 한 개만 허용한다.

Timeout과 backend error는 quality 0으로 기록하고 실제 관측 latency와 rate denominator에 포함한다.
정상 zero-result도 quality 0이며 성공 latency denominator에 포함한다. Hybrid 한 branch만 성공한
partial은 fallback으로 구분하고 성공 branch의 ranking으로 quality를 계산한다. Outlier와 실패 row는
삭제하지 않는다.

## Aggregation

Measured 결과는 overall, query tag, dataset source별로 Recall@5, Recall@10, MRR@10, nDCG@10과
p50/p95/p99, timeout/fallback/error/budget-violation rate를 계산한다. Percentile은
Hyndman–Fan type 7 linear interpolation을 사용한다.

반복 trial은 먼저 query 안에서 평균내고 query cluster를 seed 20260809로 10,000회 resample한다.
Paired comparison도 동일 query set의 차이를 resample한다. BestFixed@Budget은 validation measured
row의 Recall@10 최대값을 선택하되 p95가 budget 이하인 P4/P5/P6/P8만 feasible로 본다. 동률은
p95, depth, plan ID 순으로 결정한다. Feasible plan이 없으면 가장 낮은 p95의 best-effort plan을
기록한다. 생성된 `best_fixed_validation.json`은 test 실행에서 변경할 수 없는 lock이다.

## Output layout

```text
benchmark/results/<run_id>/
├── .run.lock
├── protocol.yaml
├── environment.json
├── run_manifest.json
├── raw_trials.jsonl
├── raw_trials.csv
├── aggregate.json
├── aggregate.csv
├── best_fixed_validation.json
└── checksums.json
```

`aggregate` 명령은 완전한 raw matrix에서 위 derived artifact를 byte-stable하게 다시 만들 수 있다.
대용량 result directory는 `.gitignore` 대상이며, 제출할 요약은 별도 검토 후에만 선별한다.

## 실행

Stage 2의 8,604개 frozen chunk를 동일 embedding/extractor로 Qdrant와 Neo4j에 적재하고 dual-store
active manifest를 만든 뒤 Stage 6 환경변수를 설정한다. 그 다음 README의
`capture-environment`, benchmark image build, `run`, `aggregate` 순서를 사용한다. Primary
runner는 Compose의 `benchmark` profile에서 Qdrant/Neo4j와 같은 local Docker network에 있고
API network round trip 없이 production engine을 직접 호출한다. Corpus 또는 backend evidence가
config와 다르면 첫 query 전에 종료한다.
