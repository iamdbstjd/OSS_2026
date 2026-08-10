# Adaptive RAG Query Optimizer
## Vector Search + Graph Retrieval Cost/Latency-Aware Hybrid Search Engine

> **문서 유형:** Product Requirements Document (PRD) + Implementation Roadmap  
> **버전:** v1.0  
> **작성 기준일:** 2026-08-09  
> **목표:** 2026 오픈소스 개발자대회 출품 가능한 오픈소스 MVP 구현 및 정량 성능 검증  
> **권장 프로젝트명(가칭):** `AdaptiveRAG` / `RAG Optimizer` / `HyRAG Optimizer`

---

# 1. 문서 목적

본 문서는 Vector Retrieval과 Knowledge Graph 기반 Graph Retrieval을 질의 특성과 latency budget에 따라 동적으로 조합하는 **Adaptive RAG Query Optimizer**를 실제 오픈소스 프로젝트로 구현하기 위한 PRD이다.

이 문서는 다음을 동시에 만족하도록 작성한다.

1. 팀원이 전체 제품의 목적과 기술적 차별점을 이해할 수 있어야 한다.
2. 각 개발 단계를 GitHub Issue/PR 단위로 분해할 수 있어야 한다.
3. 구현 범위를 명확히 제한하여 대회 마감 전에 실행 가능한 MVP를 확보해야 한다.
4. Vector-only / Graph-only / Fixed Hybrid / Adaptive Hybrid를 동일 조건에서 비교할 수 있어야 한다.
5. Accuracy/Recall과 Latency를 정량적으로 검증할 수 있어야 한다.
6. 심사위원이 `docker compose up` 수준에서 프로젝트를 재현할 수 있어야 한다.
7. 특정 Vector DB나 Graph DB에 종속되지 않는 오픈소스 미들웨어 구조를 지향해야 한다.

---

# 2. Background / Problem Statement

## 2.1 기존 Vector RAG의 문제

일반적인 Vector RAG는 질문과 문서 chunk 간 embedding similarity를 사용하여 관련 문서를 찾는다.

장점:

- 구현이 단순하다.
- semantic similarity 기반 검색에 강하다.
- ANN(Vector Approximate Nearest Neighbor) 기반으로 빠른 검색이 가능하다.
- 대부분의 RAG framework와 쉽게 통합된다.

하지만 다음 형태의 질의에서는 한계가 발생할 수 있다.

### 관계 중심 질문

```text
A 회사의 창업자와 B 프로젝트는 어떤 관계인가?
```

### Multi-hop 질문

```text
제품 X를 만든 회사의 창업자가 졸업한 대학은 어디인가?
```

### 연결 경로를 요구하는 질문

```text
Entity A와 Entity D가 어떤 관계를 통해 연결되는가?
```

Vector Search는 각 chunk의 semantic similarity를 독립적으로 계산하기 때문에, 여러 문서 또는 여러 entity 사이의 명시적 연결 관계를 탐색하는 데 적합하지 않을 수 있다.

---

## 2.2 Graph RAG의 문제

Knowledge Graph 기반 retrieval은 entity-relation 구조를 활용할 수 있다.

예:

```text
(Product X)
   |
   | CREATED_BY
   v
(Company A)
   |
   | FOUNDED_BY
   v
(Person B)
   |
   | ATTENDED
   v
(University C)
```

Graph traversal을 이용하면 multi-hop relationship query에서 강점을 가진다.

그러나 모든 query에 graph retrieval을 수행하면 다음 문제가 발생할 수 있다.

- entity extraction 비용
- graph traversal 비용
- graph depth 증가에 따른 후보 폭증
- 불필요한 Graph DB round-trip
- reranking 비용 증가
- tail latency 증가

즉, Graph Retrieval은 강력하지만 모든 query에서 동일한 수준으로 사용할 필요는 없다.

---

# 3. Product Vision

## 3.1 한 줄 정의

> **질문의 특성과 latency budget을 분석하여 Vector Search와 Graph Search의 실행계획을 자동으로 생성하고, 주어진 비용 안에서 최대 retrieval quality를 목표로 하는 오픈소스 RAG Query Optimizer**

---

## 3.2 핵심 철학

본 프로젝트는 단순히 다음을 구현하는 것이 아니다.

```text
Vector 결과 50%
+
Graph 결과 50%
```

대신 다음 문제를 해결한다.

```text
이 질문에서 Graph Search가 필요한가?

필요하다면 몇 hop까지 탐색할 것인가?

Vector 결과는 몇 개 가져올 것인가?

Graph 결과는 몇 개 가져올 것인가?

두 검색을 병렬로 실행할 것인가?

Latency budget을 넘을 경우 어떤 작업을 취소할 것인가?

Vector와 Graph 결과를 어떤 비율로 fusion할 것인가?

Reranker를 사용할 가치가 있는가?
```

이를 하나의 `RetrievalPlan`으로 표현한다.

---

# 4. 핵심 차별점

## 4.1 Query-aware Routing

Query의 특징을 분석하여 retrieval mode를 변경한다.

예:

```text
"What is Rust ownership?"
```

예상 plan:

```yaml
vector_enabled: true
graph_enabled: false
vector_top_k: 20
graph_depth: 0
rerank_enabled: false
```

반면:

```text
"Which university did the founder of company X attend?"
```

예상 plan:

```yaml
vector_enabled: true
graph_enabled: true
vector_top_k: 15
graph_top_k: 30
graph_depth: 2
vector_weight: 0.3
graph_weight: 0.7
rerank_enabled: true
```

---

## 4.2 Cost-aware Query Planning

각 candidate plan에 대해 다음 값을 추정한다.

```text
Predicted Retrieval Quality
Predicted Latency
```

그리고 latency budget을 만족하는 plan 중 quality가 가장 높다고 예측되는 plan을 선택한다.

개념:

```text
maximize    predicted_quality(plan, query)

subject to  predicted_latency(plan, query)
            <= latency_budget
```

---

## 4.3 Dynamic Graph Expansion

질의마다 graph traversal depth를 변경한다.

```text
Simple Query
→ Graph OFF 또는 depth 1

Relationship Query
→ depth 1~2

Multi-hop Query
→ depth 2~3
```

고정 depth 방식보다 불필요한 graph traversal을 줄이는 것을 목표로 한다.

---

## 4.4 Parallel Execution

Hybrid plan에서는 Vector Search와 Graph Search를 가능한 경우 동시에 실행한다.

```text
                ┌── Vector Search ──┐
Query ──────────┤                    ├── Fusion
                └── Graph Search ───┘
```

직렬:

```text
T ≈ T_vector + T_graph
```

병렬:

```text
T ≈ max(T_vector, T_graph) + fusion overhead
```

---

## 4.5 Graceful Degradation

Graph branch가 latency budget 안에 완료되지 못해도 전체 request를 실패시키지 않는다.

예:

```text
Latency Budget = 100ms

Vector Search
→ 24ms 완료

Graph Search
→ 100ms까지 완료되지 않음

결과
→ Graph task cancel
→ Vector 결과만 반환
→ fallback=true 기록
```

---

## 4.6 Storage-independent Architecture

Core engine은 특정 DB 구현에 직접 결합하지 않는다.

```text
                     AdaptiveRAG
                          |
             +------------+------------+
             |                         |
      VectorBackend               GraphBackend
             |                         |
          Qdrant                     Neo4j
          LanceDB*                   NetworkX*
```

`*`는 대회 MVP 이후 확장 범위.

---

# 5. Goals

## G1. 실행 가능한 Hybrid Retrieval Engine

하나의 API를 통해 아래 검색 전략을 실행할 수 있어야 한다.

- Vector-only
- Graph-only
- Fixed Hybrid
- Rule-based Adaptive
- Cost-aware Adaptive

---

## G2. Query Planner

Query 특징과 latency budget을 입력받아 `RetrievalPlan`을 생성할 수 있어야 한다.

---

## G3. 정량 Benchmark

최소 다음을 자동 측정할 수 있어야 한다.

Retrieval:

- Recall@5
- Recall@10
- MRR@10
- nDCG@10

System:

- p50 latency
- p95 latency
- p99 latency
- timeout rate

---

## G4. Reproducibility

신규 사용자가 README만 보고 다음을 수행할 수 있어야 한다.

```bash
git clone ...
docker compose up -d
pip install -e .
python scripts/ingest.py ...
python scripts/demo.py ...
python scripts/benchmark.py ...
```

---

## G5. Explainability

각 query에서 다음 내용을 확인할 수 있어야 한다.

```text
왜 이 RetrievalPlan이 선택되었는가?
예상 latency는 얼마였는가?
실제 latency는 얼마였는가?
Vector와 Graph가 각각 얼마나 걸렸는가?
fallback이 발생했는가?
```

---

# 6. Non-Goals

대회 MVP에서는 아래를 핵심 범위에 포함하지 않는다.

## NG1. 자체 Vector DB 개발

Qdrant를 사용한다.

## NG2. 자체 Graph DB 개발

Neo4j를 사용한다.

## NG3. 자체 Embedding Model 학습

기존 embedding model을 사용한다.

## NG4. 자체 LLM 학습

필요하면 외부 또는 공개 모델을 사용한다.

## NG5. 모든 Vector/Graph DB 지원

MVP:

```text
Vector = Qdrant
Graph  = Neo4j
```

추후 adapter만 확장한다.

## NG6. React Dashboard

대회 MVP는 CLI + REST API를 우선한다.

## NG7. 전체 Rust Rewrite

Python-first로 구현한다.

실제 profiling에서 Python 코드가 병목이라는 증거가 있을 때만 Rust extension을 검토한다.

## NG8. Agent Framework

본 프로젝트의 핵심은 Retrieval Optimization이며 Agent는 범위 밖이다.

---

# 7. 사용자

## Persona A. RAG Application Developer

요구:

```text
내가 직접 Vector/Graph weight를 튜닝하고 싶지 않다.
Query마다 자동으로 좋은 검색 전략을 사용하고 싶다.
```

---

## Persona B. AI Infra Engineer

요구:

```text
Latency SLA 안에서 최대한 좋은 retrieval quality를 유지하고 싶다.
각 retrieval 단계의 latency trace를 보고 싶다.
```

---

## Persona C. RAG Researcher

요구:

```text
Vector-only, Graph-only, Hybrid를 동일 데이터에서 비교하고 싶다.
Planner를 교체하며 benchmark하고 싶다.
```

---

# 8. 핵심 User Stories

## US-001

사용자로서 query를 입력하면 별도의 retrieval strategy 설정 없이 검색 결과를 받고 싶다.

Acceptance Criteria:

- `/v1/search` 요청 가능
- 기본 planner가 자동 적용
- 검색 결과 반환
- 선택된 plan이 response에 포함

---

## US-002

사용자로서 latency budget을 지정하고 싶다.

예:

```json
{
  "query": "...",
  "latency_budget_ms": 150
}
```

Acceptance Criteria:

- budget 값이 optimizer에 전달
- budget에 따라 선택 plan이 달라질 수 있음
- 실제 latency가 trace에 기록
- timeout branch가 존재

---

## US-003

연구자로서 Vector-only benchmark를 실행하고 싶다.

```bash
adaptive-rag benchmark --method vector
```

Acceptance Criteria:

- deterministic benchmark config 저장
- metric JSON/CSV 생성
- latency raw data 저장

---

## US-004

연구자로서 모든 plan을 동일 query에 실행하여 oracle plan을 찾고 싶다.

Acceptance Criteria:

- query × plan matrix 생성
- 각 row에 quality 및 latency 포함
- offline training dataset으로 사용 가능

---

# 9. System Architecture

```text
                       ┌──────────────────┐
                       │      Client      │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   REST / Python  │
                       │       SDK        │
                       └────────┬─────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │     Query Analyzer   │
                    │                      │
                    │ token count          │
                    │ entities             │
                    │ relation signal      │
                    │ multi-hop signal     │
                    │ semantic features    │
                    └───────────┬──────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │    Query Optimizer   │
                    │                      │
                    │ Plan Space           │
                    │ Quality Model        │
                    │ Latency Model        │
                    │ Budget Constraint    │
                    └───────────┬──────────┘
                                │
                                ▼
                         RetrievalPlan
                                │
                                ▼
                    ┌──────────────────────┐
                    │ Execution Scheduler  │
                    └───────────┬──────────┘
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
        ┌───────────────┐                ┌───────────────┐
        │ VectorBackend │                │ GraphBackend  │
        │    Qdrant     │                │     Neo4j     │
        └───────┬───────┘                └───────┬───────┘
                │                                │
                └───────────────┬────────────────┘
                                ▼
                       ┌──────────────────┐
                       │ Hybrid Fusion    │
                       │ Weighted RRF     │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Optional Rerank  │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Retrieval Result │
                       └──────────────────┘
```

---

# 10. Recommended Repository Structure

```text
adaptive-rag/
│
├── README.md
├── LICENSE
├── THIRD_PARTY_LICENSES.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── docker-compose.yml
├── Makefile
├── .env.example
├── .gitignore
│
├── src/
│   └── adaptive_rag/
│       ├── api/
│       │   ├── server.py
│       │   ├── routes.py
│       │   └── schemas.py
│       │
│       ├── core/
│       │   ├── engine.py
│       │   ├── types.py
│       │   ├── config.py
│       │   └── exceptions.py
│       │
│       ├── backends/
│       │   ├── vector/
│       │   │   ├── base.py
│       │   │   └── qdrant.py
│       │   └── graph/
│       │       ├── base.py
│       │       └── neo4j.py
│       │
│       ├── planner/
│       │   ├── analyzer.py
│       │   ├── features.py
│       │   ├── plans.py
│       │   ├── rule_planner.py
│       │   ├── optimizer.py
│       │   ├── quality_model.py
│       │   └── latency_model.py
│       │
│       ├── retrieval/
│       │   ├── vector.py
│       │   ├── graph.py
│       │   ├── fusion.py
│       │   └── reranker.py
│       │
│       ├── scheduler/
│       │   ├── executor.py
│       │   ├── timeout.py
│       │   └── budget.py
│       │
│       ├── ingestion/
│       │   ├── models.py
│       │   ├── chunker.py
│       │   ├── embedder.py
│       │   ├── entities.py
│       │   ├── relations.py
│       │   ├── qdrant_writer.py
│       │   ├── neo4j_writer.py
│       │   └── pipeline.py
│       │
│       └── observability/
│           ├── trace.py
│           ├── metrics.py
│           └── logger.py
│
├── benchmark/
│   ├── README.md
│   ├── datasets/
│   ├── configs/
│   ├── runners/
│   ├── metrics/
│   ├── analysis/
│   └── results/
│
├── scripts/
│   ├── ingest.py
│   ├── demo.py
│   ├── benchmark.py
│   ├── profile_plans.py
│   └── train_cost_models.py
│
├── examples/
│   ├── basic_search.py
│   ├── adaptive_search.py
│   └── latency_budget.py
│
└── tests/
    ├── unit/
    ├── integration/
    └── benchmark/
```

---

# 11. Core Domain Models

## 11.1 Chunk

```python
@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    position: int
    entities: list[str]
    metadata: dict
```

---

## 11.2 RetrievalHit

```python
@dataclass
class RetrievalHit:
    id: str
    text: str
    score: float
    source: str
    document_id: str | None = None
    entity_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

---

## 11.3 GraphRetrievalHit

```python
@dataclass
class GraphRetrievalHit(RetrievalHit):
    path: list[str] = field(default_factory=list)
    hop_count: int = 0
```

---

## 11.4 QueryFeatures

```python
@dataclass
class QueryFeatures:
    token_count: int
    entity_count: int
    entity_density: float

    relation_signal: float
    multi_hop_signal: float
    comparison_signal: float
    aggregation_signal: float
    global_signal: float

    query_embedding: list[float] | None = None
```

---

## 11.5 RetrievalPlan

프로젝트에서 가장 중요한 domain object.

```python
@dataclass
class RetrievalPlan:
    id: str

    vector_enabled: bool
    graph_enabled: bool

    vector_top_k: int
    graph_top_k: int

    graph_depth: int

    vector_weight: float
    graph_weight: float

    rerank_enabled: bool
    rerank_top_k: int

    timeout_ms: int

    expected_quality: float | None = None
    expected_latency_ms: float | None = None
```

---

## 11.6 SearchTrace

```python
@dataclass
class SearchTrace:
    request_id: str

    query: str
    query_features: QueryFeatures

    selected_plan: RetrievalPlan

    analyzer_latency_ms: float
    planner_latency_ms: float

    vector_latency_ms: float | None
    graph_latency_ms: float | None
    fusion_latency_ms: float | None
    rerank_latency_ms: float | None

    total_latency_ms: float

    timeout: bool
    fallback: bool

    result_count: int
```

---

# 12. Backend Interfaces

## 12.1 VectorBackend

```python
class VectorBackend(Protocol):

    async def search(
        self,
        query: str,
        top_k: int,
        timeout_ms: int | None = None,
    ) -> list[RetrievalHit]:
        ...
```

Qdrant 구현:

```text
QdrantVectorBackend
```

---

## 12.2 GraphBackend

```python
class GraphBackend(Protocol):

    async def search(
        self,
        query: str,
        seed_entities: list[str],
        top_k: int,
        depth: int,
        timeout_ms: int | None = None,
    ) -> list[GraphRetrievalHit]:
        ...
```

Neo4j 구현:

```text
Neo4jGraphBackend
```

---

# 13. Storage Data Model

## 13.1 Qdrant

Qdrant payload 예:

```json
{
  "id": "wiki_rust_001_03",
  "vector": [0.123, 0.234],
  "payload": {
    "document_id": "wiki_rust_001",
    "text": "...",
    "position": 3,
    "entities": [
      "Rust",
      "Mozilla"
    ]
  }
}
```

---

## 13.2 Neo4j

Graph schema:

```text
(Document)
    |
    | HAS_CHUNK
    v
(Chunk)
    |
    | MENTIONS
    v
(Entity)
    |
    | RELATION_TYPE
    v
(Entity)
```

Example:

```text
(:Document {id})
(:Chunk {id, document_id, text, position})
(:Entity {id, name, type})
```

Relations:

```text
(:Document)-[:HAS_CHUNK]->(:Chunk)

(:Chunk)-[:MENTIONS]->(:Entity)

(:Entity)-[:RELATES_TO {
    type,
    confidence
}]->(:Entity)
```

---

## 13.3 ID Consistency

Qdrant와 Neo4j에서 동일 chunk ID를 사용한다.

```text
wiki_rust_001_03
```

이 규칙은 필수이다.

이유:

- Vector 결과와 Graph 결과 deduplication 용이
- Fusion 단순화
- benchmark ground truth 연결 용이
- debugging 용이

---

# 14. Ingestion Pipeline

```text
Raw Document
     |
     v
Normalize
     |
     v
Chunking
     |
     +----------------------+
     |                      |
     v                      v
Embedding             Entity Extraction
     |                      |
     v                      v
Qdrant              Relation Extraction
                            |
                            v
                          Neo4j
```

---

# 15. Implementation Phase 0 — Project Bootstrap

## 목적

팀이 동일한 환경에서 바로 개발을 시작할 수 있도록 기본 repository와 local infrastructure를 만든다.

## 구현 항목

### P0-001 Repository 생성

파일:

```text
README.md
LICENSE
.gitignore
pyproject.toml
```

### P0-002 Docker Compose

서비스:

```text
qdrant
neo4j
adaptive-rag-api
```

### P0-003 환경설정

```text
.env.example
```

예:

```env
QDRANT_URL=http://localhost:6333

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### P0-004 CI

최소:

```text
ruff
pytest
```

## Definition of Done

```bash
docker compose up -d
```

후:

- Qdrant health 확인
- Neo4j 연결 확인
- `pytest` 통과
- sample API `/health` 200

---

# 16. Implementation Phase 1 — Vector Baseline

## 목적

가장 단순하고 안정적인 baseline을 확보한다.

## Pipeline

```text
Query
 ↓
Embedding
 ↓
Qdrant Search
 ↓
Top-K Chunks
```

## 개발 Task

### VEC-001 Embedder Interface

```python
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...
```

### VEC-002 Qdrant Collection 초기화

필수 설정:

```text
collection name
embedding dimension
distance metric
```

### VEC-003 Chunk Upsert

입력:

```text
Chunk[]
```

결과:

```text
Qdrant points
```

### VEC-004 Search

```python
await vector_backend.search(
    query,
    top_k=10,
)
```

### VEC-005 Latency Instrumentation

최소:

```text
embedding latency
qdrant latency
total vector latency
```

## Acceptance Criteria

아래 명령이 성공해야 한다.

```bash
python scripts/demo.py \
  --mode vector \
  --query "What is Rust ownership?"
```

출력 예:

```text
MODE      VECTOR

1 chunk_x
2 chunk_y
3 chunk_z

Embedding     14.2 ms
Qdrant         8.3 ms
Total         23.1 ms
```

## Tests

Unit:

- embedding output shape
- result serialization
- top_k validation

Integration:

- Qdrant insert
- Qdrant query
- empty collection
- nonexistent collection

---

# 17. Implementation Phase 2 — Knowledge Graph Ingestion

## 목적

문서 chunk와 entity/relation graph를 생성한다.

## Task

### KG-001 Entity Model

```python
@dataclass
class Entity:
    id: str
    name: str
    type: str | None
```

### KG-002 Relation Model

```python
@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: str
    confidence: float | None
```

### KG-003 Entity Extraction

초기에는 복잡한 자체 NER model을 연구하지 않는다.

목표:

```text
document/chunk
→ entities
```

### KG-004 Relation Extraction

목표:

```text
chunk
→ subject
→ relation
→ object
```

### KG-005 Neo4j Writer

생성:

```text
Document
Chunk
Entity
HAS_CHUNK
MENTIONS
Entity Relation
```

### KG-006 Deduplication

동일 entity를 가능한 범위에서 정규화한다.

예:

```text
OpenAI
openai
Open AI
```

단, entity resolution 연구 자체에 과도한 시간을 쓰지 않는다.

---

## Acceptance Criteria

Neo4j Browser 또는 query를 통해:

```cypher
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
RETURN d, c
LIMIT 10
```

동작.

그리고:

```cypher
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN c, e
LIMIT 10
```

동작.

---

# 18. Implementation Phase 3 — Graph Retrieval Baseline

## 목적

Graph-only retrieval을 독립적으로 실행할 수 있게 한다.

## Pipeline

```text
Query
 ↓
Seed Entity Extraction
 ↓
Entity Match
 ↓
Graph Expansion
 ↓
Related Entities
 ↓
Associated Chunks
 ↓
Ranking
 ↓
Top-K
```

## Task

### GRAPH-001 Query Entity Extraction

질의에서 seed entity 후보를 추출.

### GRAPH-002 Entity Lookup

Neo4j의 Entity node와 연결.

### GRAPH-003 Graph Expansion

초기 지원:

```text
depth = 1
depth = 2
depth = 3
```

### GRAPH-004 Chunk Recovery

탐색된 entity와 연결된 chunk를 반환.

### GRAPH-005 Graph Score

MVP에서는 간단한 heuristic으로 시작한다.

예:

```text
shorter path
→ higher score

more seed overlap
→ higher score
```

## Graph Query 개념 예

실제 Cypher는 데이터 크기와 schema에 맞게 최적화한다.

```cypher
MATCH (seed:Entity)
WHERE seed.name IN $entities

MATCH path=(seed)-[*1..2]-(related:Entity)

MATCH (chunk:Chunk)-[:MENTIONS]->(related)

RETURN chunk, path
LIMIT $top_k
```

주의:

- 무제한 `[*]` 금지
- depth 제한
- 실제 benchmark 전에 `EXPLAIN`/`PROFILE` 검토

---

## Acceptance Criteria

```bash
python scripts/demo.py \
  --mode graph \
  --query "Where did the founder of X study?"
```

가 동작.

Response에:

```text
seed entities
graph depth
graph paths
related chunks
latency
```

가 포함.

---

# 19. Implementation Phase 4 — Fixed Hybrid Baseline

## 목적

Adaptive Planner가 실제 가치가 있는지 비교할 기준을 만든다.

## Baseline

```text
Vector-only
Graph-only
Fixed Hybrid
```

Fixed Hybrid 기본:

```text
vector_weight = 0.5
graph_weight = 0.5
```

---

## 19.1 Fusion

MVP 추천:

**Weighted Reciprocal Rank Fusion**

개념:

```text
score(d)
=
vector_weight / (k + vector_rank)
+
graph_weight / (k + graph_rank)
```

Python 예:

```python
def weighted_rrf(
    vector_results,
    graph_results,
    vector_weight,
    graph_weight,
    k=60,
):
    scores = {}

    for rank, item in enumerate(vector_results, 1):
        scores[item.id] = (
            scores.get(item.id, 0.0)
            + vector_weight / (k + rank)
        )

    for rank, item in enumerate(graph_results, 1):
        scores[item.id] = (
            scores.get(item.id, 0.0)
            + graph_weight / (k + rank)
        )

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
```

## RRF를 초기 방식으로 선택하는 이유

Vector score와 Graph score는 서로 다른 의미와 scale을 가질 수 있다.

예:

```text
Vector cosine similarity 0.81

Graph heuristic score 3.2
```

이를 그대로 선형 합산하기보다 rank 기반 fusion을 사용하는 것이 초기 구현에 안전하다.

---

## Acceptance Criteria

아래 세 모드가 동일 CLI에서 실행 가능.

```bash
python scripts/benchmark.py --method vector

python scripts/benchmark.py --method graph

python scripts/benchmark.py --method fixed-hybrid
```

---

# 20. Implementation Phase 5 — Benchmark Harness V0

## 중요도

**최우선 P0.**

Planner 구현보다 먼저 baseline benchmark가 있어야 한다.

## 목적

모든 실험을 동일 조건에서 재현한다.

## Benchmark Record

```python
@dataclass
class BenchmarkRecord:
    query_id: str
    method: str
    plan_id: str | None

    recall_5: float
    recall_10: float
    mrr_10: float
    ndcg_10: float

    latency_ms: float

    timeout: bool
```

---

## Raw Result 저장

CSV:

```csv
query_id,method,plan_id,recall10,mrr10,latency_ms
q001,vector,,0.50,0.42,21.3
q001,graph,,0.75,0.70,81.2
q001,fixed,,1.00,0.92,73.4
```

그리고 aggregate JSON:

```json
{
  "method": "vector",
  "queries": 500,
  "recall_at_10": 0.73,
  "p50_latency_ms": 24,
  "p95_latency_ms": 39
}
```

---

## Dataset Split

Planner 학습/튜닝 dataset과 최종 evaluation dataset을 반드시 분리한다.

예:

```text
Train       60%
Validation  20%
Test        20%
```

원칙:

```text
Test set은 최종 수치 측정 전까지 parameter tuning에 사용하지 않는다.
```

---

# 21. Implementation Phase 6 — Query Analyzer V0

## 목적

Query를 Planner가 사용할 수 있는 숫자 feature로 변환한다.

## 초기 Feature Set

```text
token_count
entity_count
entity_density
relation_signal
multi_hop_signal
comparison_signal
aggregation_signal
global_signal
```

---

## 관계 Signal 예

관계 중심 표현:

```text
founded by
created by
works at
connected to
relationship
between
parent
subsidiary
```

한국어 dataset을 사용한다면 한국어 pattern도 추가한다.

---

## Multi-hop Signal 예

```text
"the founder of the company that..."
```

```text
"X를 만든 회사의 창업자가..."
```

chain expression이 증가할수록 multi-hop signal을 높인다.

---

## LLM Router를 P0에서 제외하는 이유

온라인 Planner 앞에서 LLM 호출을 수행하면:

- planner latency 증가
- 외부 API 종속
- benchmark noise 증가
- 비용 발생
- 오프라인 재현성 감소

따라서 V0 Analyzer는 lightweight 방식으로 구현한다.

추후 실험적으로 LLM analyzer를 adapter로 추가할 수 있다.

---

## Acceptance Criteria

```python
features = analyzer.analyze(
    "Where did the founder of X study?"
)
```

결과 예:

```json
{
  "entity_count": 1,
  "relation_signal": 0.91,
  "multi_hop_signal": 0.88
}
```

---

# 22. Implementation Phase 7 — Rule-based Adaptive Planner V0

## 목적

ML cost model이 없어도 adaptive behavior를 구현한다.

## Plan Presets

### P0 — VECTOR_FAST

```yaml
id: P0

vector_enabled: true
graph_enabled: false

vector_top_k: 10
graph_top_k: 0

graph_depth: 0

vector_weight: 1.0
graph_weight: 0.0

rerank_enabled: false
```

### P1 — VECTOR_WIDE

```yaml
id: P1

vector_enabled: true
graph_enabled: false

vector_top_k: 30
```

### P2 — GRAPH_SHALLOW

```yaml
id: P2

vector_enabled: false
graph_enabled: true

graph_top_k: 20
graph_depth: 1
```

### P3 — GRAPH_DEEP

```yaml
id: P3

graph_enabled: true

graph_top_k: 30
graph_depth: 2
```

### P4 — HYBRID_VECTOR_HEAVY

```yaml
id: P4

vector_enabled: true
graph_enabled: true

vector_top_k: 20
graph_top_k: 15

graph_depth: 1

vector_weight: 0.7
graph_weight: 0.3
```

### P5 — HYBRID_BALANCED

```yaml
id: P5

vector_top_k: 20
graph_top_k: 20

graph_depth: 1

vector_weight: 0.5
graph_weight: 0.5
```

### P6 — HYBRID_GRAPH_HEAVY

```yaml
id: P6

vector_top_k: 15
graph_top_k: 30

graph_depth: 2

vector_weight: 0.3
graph_weight: 0.7
```

### P7 — HYBRID_RERANK

```yaml
id: P7

vector_top_k: 30
graph_top_k: 30

graph_depth: 2

vector_weight: 0.5
graph_weight: 0.5

rerank_enabled: true
rerank_top_k: 10
```

---

## Rule Example

```python
def choose_plan(features, budget_ms):

    if budget_ms <= 50:
        return P0

    if features.multi_hop_signal >= 0.8:
        return P6

    if features.relation_signal >= 0.7:
        return P5

    return P1
```

---

## Acceptance Criteria

동일 query라도 budget 변경 시 plan이 바뀔 수 있어야 한다.

```text
50ms  → P0
100ms → P4
300ms → P6
```

정확한 threshold는 benchmark 결과로 조정.

---

# 23. Implementation Phase 8 — Offline Plan Profiler

## 목적

각 query에서 어떤 plan이 실제로 가장 좋은지 측정한다.

## 방법

모든 benchmark query에 대해 모든 candidate plan 실행.

```text
Q1 × P0
Q1 × P1
...
Q1 × P7

Q2 × P0
...
```

결과:

```csv
query_id,plan_id,recall10,latency_ms
q001,P0,0.50,21
q001,P1,0.75,29
q001,P2,0.75,61
q001,P3,1.00,118
q001,P4,1.00,72
```

---

## Offline Training Dataset

Feature까지 합친다.

```csv
query_id,
token_count,
entity_count,
relation_signal,
multi_hop_signal,
plan_id,
graph_depth,
vector_top_k,
graph_top_k,
vector_weight,
graph_weight,
recall10,
latency_ms
```

---

## Acceptance Criteria

명령:

```bash
python scripts/profile_plans.py \
  --dataset benchmark/configs/train.yaml
```

출력:

```text
benchmark/results/profile_train.csv
```

---

# 24. Implementation Phase 9 — Cost Models

## 목적

Query를 실행하기 전에 candidate plan의 예상 quality/latency를 추정한다.

---

## 24.1 Quality Model

입력:

```text
QueryFeatures
+
Plan Features
```

출력:

```text
Predicted Recall@10
```

---

## 24.2 Latency Model

입력:

```text
QueryFeatures
+
Plan Features
```

출력:

```text
Predicted Latency
```

---

## Model 후보

P0에서는 다음 정도로 충분하다.

```text
Linear Regression
Random Forest
Gradient Boosting
```

복잡한 neural model은 필수가 아니다.

핵심은 model complexity가 아니라:

```text
offline profiling
→ cost estimation
→ plan selection
```

구조에 있다.

---

## Plan Feature 예

```text
vector_enabled
graph_enabled

vector_top_k
graph_top_k

graph_depth

rerank_enabled
```

---

## Acceptance Criteria

```python
quality_hat = quality_model.predict(
    query_features,
    plan,
)

latency_hat = latency_model.predict(
    query_features,
    plan,
)
```

동작.

Validation set에서 prediction error 측정.

Latency model:

```text
MAE
RMSE
```

Quality model:

```text
MAE
RMSE
또는 ranking accuracy
```

기록.

---

# 25. Implementation Phase 10 — Cost-aware Query Optimizer

## 핵심 알고리즘

```python
def optimize(
    query_features,
    latency_budget_ms,
):
    feasible = []

    for plan in PLAN_SPACE:

        quality = quality_model.predict(
            query_features,
            plan,
        )

        latency = latency_model.predict(
            query_features,
            plan,
        )

        if latency <= latency_budget_ms:
            feasible.append(
                (
                    plan,
                    quality,
                    latency,
                )
            )

    if not feasible:
        return FASTEST_SAFE_PLAN

    return max(
        feasible,
        key=lambda x: x[1],
    )
```

---

## Planner 결과

```json
{
  "selected_plan": "P6",

  "expected_quality": 0.86,

  "expected_latency_ms": 82,

  "latency_budget_ms": 100
}
```

---

## Acceptance Criteria

1. 모든 candidate plan을 scoring할 수 있음.
2. budget 초과 predicted plan을 제외함.
3. feasible plan 중 최대 predicted quality plan 선택.
4. feasible plan이 없으면 fallback plan 선택.
5. 선택 이유가 trace에 기록됨.

---

# 26. Implementation Phase 11 — Async Execution Scheduler

## 목적

Vector와 Graph branch를 병렬 실행한다.

## Example

```python
vector_task = asyncio.create_task(
    vector_backend.search(...)
)

graph_task = asyncio.create_task(
    graph_backend.search(...)
)
```

---

## Scheduler State

```text
START
 |
 +→ Vector RUNNING
 |
 +→ Graph RUNNING
 |
 +→ budget monitor
```

---

## Timeout

```python
try:
    result = await asyncio.wait_for(
        task,
        timeout=remaining_seconds,
    )
except asyncio.TimeoutError:
    ...
```

---

## Acceptance Criteria

Hybrid query에서:

```text
vector_latency = 30ms
graph_latency  = 80ms
```

일 때 전체 retrieval 시간이 단순 합인 110ms에 가깝게 직렬 수행되지 않고, 병렬 수행 효과가 나타나야 한다.

실제 수치는 환경에 따라 달라지므로 benchmark로 측정한다.

---

# 27. Implementation Phase 12 — Graceful Degradation

## 목적

한 branch의 지연이 전체 query failure로 이어지지 않게 한다.

## 정책

### Case A

Vector 완료, Graph timeout.

```text
→ Vector result 반환
```

### Case B

Graph 완료, Vector timeout.

```text
→ Graph result 반환
```

### Case C

둘 다 실패.

```text
→ RetrievalError
```

### Case D

Fusion 전에 budget 거의 소진.

```text
→ reranker skip
```

---

## Trace Example

```json
{
  "plan": "P6",

  "budget_ms": 100,

  "vector": {
    "status": "success",
    "latency_ms": 24.1
  },

  "graph": {
    "status": "timeout",
    "latency_ms": 74.8
  },

  "fallback": true,

  "total_latency_ms": 100.3
}
```

---

# 28. Implementation Phase 13 — Reranker

## 우선순위

P1.

P0 core benchmark가 확보된 후 추가한다.

## 조건부 실행

Planner가 다음 plan을 선택한 경우에만 실행:

```text
P7
```

또는:

```text
remaining latency budget 충분
```

---

## Acceptance Criteria

reranker ON/OFF 비교 가능.

Ablation에서:

```text
Full
vs
No Reranker
```

비교.

---

# 29. REST API PRD

## POST /v1/search

Request:

```json
{
  "query": "Where did the founder of X study?",
  "top_k": 10,
  "latency_budget_ms": 150,

  "planner": "adaptive"
}
```

Response:

```json
{
  "query": "Where did the founder of X study?",

  "results": [
    {
      "id": "chunk_001",
      "text": "...",
      "score": 0.91,
      "source": "hybrid"
    }
  ],

  "planner": {
    "type": "adaptive",

    "features": {
      "entity_count": 1,
      "relation_signal": 0.91,
      "multi_hop_signal": 0.88
    },

    "selected_plan": {
      "id": "P6",

      "vector_top_k": 15,
      "graph_top_k": 30,

      "graph_depth": 2,

      "vector_weight": 0.3,
      "graph_weight": 0.7
    },

    "expected_latency_ms": 84
  },

  "execution": {
    "vector_latency_ms": 23,
    "graph_latency_ms": 68,
    "fusion_latency_ms": 1,

    "total_latency_ms": 72,

    "fallback": false
  }
}
```

---

## GET /health

```json
{
  "status": "ok",

  "qdrant": "ok",
  "neo4j": "ok"
}
```

---

## GET /metrics

선택적으로 Prometheus-compatible format 고려.

MVP에서는 JSON도 허용.

---

# 30. CLI PRD

## Search

```bash
adaptive-rag search \
  "Where did the founder of X study?" \
  --budget 150
```

출력:

```text
Query Analysis
──────────────────────────

Entity count       1
Relation signal    0.91
Multi-hop signal   0.88


Selected Plan
──────────────────────────

Plan               P6
Vector top-k       15
Graph top-k        30
Graph depth        2

Vector weight      0.30
Graph weight       0.70

Expected latency   84 ms


Execution
──────────────────────────

Vector             23 ms
Graph              68 ms
Fusion              1 ms
Total              72 ms
```

---

## Benchmark

```bash
adaptive-rag benchmark \
  --method adaptive \
  --config benchmark/configs/test.yaml
```

---

## Profile

```bash
adaptive-rag profile-plans \
  --config benchmark/configs/train.yaml
```

---

# 31. Benchmark Strategy

## 31.1 Baselines

반드시 동일 dataset에서 비교한다.

```text
B1 Vector-only

B2 Graph-only

B3 Fixed Hybrid

B4 Rule-based Adaptive

B5 Cost-aware Adaptive
```

---

## 31.2 Metrics

### Retrieval Quality

```text
Recall@5
Recall@10
MRR@10
nDCG@10
```

### System Performance

```text
p50 latency
p95 latency
p99 latency
timeout rate
```

### Optional

```text
CPU
memory
QPS
```

---

# 32. Primary Success Metrics

대회 발표에서는 너무 많은 숫자를 한 화면에 보여주지 않는다.

Primary:

```text
Recall@10
p95 Latency
```

Secondary:

```text
MRR@10
nDCG@10
p50
p99
```

---

# 33. Benchmark Tables

## Main Table

| Method | Recall@10 | MRR@10 | p50 | p95 |
|---|---:|---:|---:|---:|
| Vector-only | measured | measured | measured | measured |
| Graph-only | measured | measured | measured | measured |
| Fixed Hybrid | measured | measured | measured | measured |
| Rule Adaptive | measured | measured | measured | measured |
| **Cost-aware Adaptive** | **measured** | **measured** | **measured** | **measured** |

절대로 측정 전 임의 숫자를 채우지 않는다.

---

## Query Type Table

| Query Type | Vector | Graph | Fixed | Adaptive |
|---|---:|---:|---:|---:|
| Semantic | | | | |
| Entity | | | | |
| Relationship | | | | |
| 2-hop | | | | |
| 3-hop | | | | |

---

# 34. Latency Budget Experiment

Budget 후보:

```text
50ms
100ms
200ms
500ms
```

각 budget에서:

```text
selected plan
Recall@10
actual latency
timeout rate
```

측정.

목표:

> Latency budget 변화에 따라 optimizer가 다른 Pareto point를 선택하는 것을 증명.

---

# 35. Pareto Analysis

최종 보고서에서 중요한 그래프.

```text
Quality
  ^
  |
  |                    Adaptive
  |                 ●
  |
  |            ● Fixed Hybrid
  |
  |     ● Vector
  |
  |                        ● Graph
  +--------------------------------> Latency
```

실제 그래프는 측정 결과를 사용한다.

핵심 메시지:

```text
동일 latency에서 더 높은 quality
또는
동일 quality에서 더 낮은 latency
```

둘 중 하나를 정량적으로 증명한다.

---

# 36. Ablation Study

## Full

```text
Query Analyzer
+
Cost Model
+
Dynamic Graph Depth
+
Parallel Scheduler
+
Timeout/Fallback
```

## Ablation

```text
A1 - Query Analyzer

A2 - Cost Model

A3 - Dynamic Graph Depth

A4 - Parallel Execution

A5 - Reranker
```

결과:

| Config | Recall@10 | p95 | Timeout |
|---|---:|---:|---:|
| Full | | | |
| - Cost Model | | | |
| - Dynamic Depth | | | |
| - Parallel | | | |
| - Reranker | | | |

---

# 37. Synthetic Graph Benchmark

실제 dataset만으로 graph depth 효과가 명확하지 않을 경우 controlled benchmark를 생성한다.

예 데이터:

```text
Alice founded Company A.

Company A acquired Company B.

Company B created Product C.

Product C uses Technology D.
```

질문:

```text
Who founded the company connected
through acquisition to Product C?
```

Expected path:

```text
Alice
→ Company A
→ Company B
→ Product C
```

질의 유형:

```text
1-hop
2-hop
3-hop
4-hop
```

주의:

Synthetic benchmark는 main benchmark를 대체하지 않는다.

보조 실험으로 사용한다.

---

# 38. Test Strategy

## Unit Tests

### Planner

```text
test_plan_for_low_budget

test_plan_for_multihop_query

test_plan_fallback
```

### Fusion

```text
test_rrf_rank

test_duplicate_chunk

test_empty_vector_result

test_empty_graph_result
```

### Scheduler

```text
test_parallel_execution

test_graph_timeout

test_vector_timeout

test_both_timeout
```

---

## Integration Tests

실제:

```text
Qdrant
Neo4j
```

를 Docker에서 구동.

테스트:

```text
ingest document
search vector
search graph
hybrid search
adaptive search
```

---

## Regression Tests

대표 query 20~50개 고정.

PR마다:

```text
result quality
latency catastrophic regression
```

확인.

Latency는 CI 환경 변동이 크므로 strict millisecond assertion보다 넓은 upper-bound 또는 dedicated benchmark 환경을 사용한다.

---

# 39. Observability

모든 query를 JSONL로 저장 가능하게 한다.

```text
logs/search_trace.jsonl
```

Example:

```json
{
  "request_id": "req_001",

  "query": "...",

  "plan": "P6",

  "vector_latency_ms": 22.3,
  "graph_latency_ms": 61.4,
  "fusion_latency_ms": 0.8,

  "total_latency_ms": 63.0,

  "fallback": false
}
```

---

# 40. Configuration

예:

```yaml
engine:

  default_latency_budget_ms: 200

planner:

  type: cost_aware

  plan_space:
    - P0
    - P1
    - P2
    - P3
    - P4
    - P5
    - P6
    - P7

vector:

  backend: qdrant
  collection: adaptive_rag

graph:

  backend: neo4j

fusion:

  method: weighted_rrf
  rrf_k: 60
```

---

# 41. Security / Privacy

MVP 수준에서도 다음을 준수한다.

## SEC-001 Secrets

API key와 password는 repository에 commit하지 않는다.

`.env.example`만 제공.

## SEC-002 Query Logging

실제 사용자 query에는 개인정보가 있을 수 있다.

따라서 production 사용을 고려하면 trace logging을 disable 또는 redact할 수 있어야 한다.

## SEC-003 Cypher Injection

사용자 자연어를 직접 문자열 interpolation하여 Cypher에 넣지 않는다.

parameterized query 사용.

---

# 42. Open Source Requirements

필수 파일:

```text
LICENSE
README.md
CONTRIBUTING.md
THIRD_PARTY_LICENSES.md
```

선택:

```text
CODE_OF_CONDUCT.md
SECURITY.md
```

Dependencies 및 dataset의 라이선스를 별도로 기록한다.

---

# 43. Performance Engineering Rules

## Rule 1

Planner 자체 latency를 반드시 측정한다.

Optimizer가 30ms 걸리고 retrieval을 10ms 줄이면 의미가 없다.

목표:

```text
Analyzer + Planner overhead
≪ Retrieval latency
```

---

## Rule 2

DB tuning과 Planner 효과를 분리한다.

초기 benchmark:

```text
default DB settings
```

후:

```text
DB tuning experiment
```

을 별도 수행.

---

## Rule 3

Cold/Warm benchmark 구분.

가능하면:

```text
cold run
warm run
```

구분.

---

## Rule 4

p95를 반드시 본다.

평균 latency만 보고하지 않는다.

---

# 44. Python vs Rust Decision

## MVP

```text
Python
```

권장.

이유:

```text
개발 속도
AI ecosystem
Qdrant/Neo4j client 지원
benchmark tooling
빠른 iteration
```

---

## Rust 도입 조건

프로파일 결과 다음과 같은 Python-side bottleneck이 실제로 확인될 때만 도입한다.

```text
fusion
serialization
ranking
scheduler overhead
```

Rust 후보:

```text
Weighted RRF
Ranking
Score normalization
```

PyO3 extension 방식 고려.

반대로 실제 latency 대부분이:

```text
Qdrant network
Neo4j traversal
Embedding
Reranker
```

에서 발생하면 Rust rewrite 우선순위가 낮다.

---

# 45. Detailed Schedule

> 기준일: 2026-08-09  
> 개발 실질 마감 목표: 2026-08-26  
> 제출 전날까지 기능 변경을 중단하고 재현성/문서/영상에 집중한다.

| 날짜 | Epic | 완료 조건 |
|---|---|---|
| 8/9 | Bootstrap | Repo + Docker + Health |
| 8/10 | Vector | Vector baseline 작동 |
| 8/11 | KG Ingestion | Document/Chunk/Entity 생성 |
| 8/12 | Graph | Graph-only search |
| 8/13 | Hybrid | Fixed Hybrid + RRF |
| 8/14 | Benchmark V0 | baseline metric 자동 측정 |
| 8/15 | Query Analyzer | QueryFeatures 생성 |
| 8/16 | Rule Planner | Adaptive V0 |
| 8/17 | Plan Profiler | query × plan CSV |
| 8/18 | Cost Models | quality/latency prediction |
| 8/19 | Optimizer | budget-aware plan selection |
| 8/20 | Scheduler | parallel + timeout |
| 8/21 | Benchmark | main dataset 1차 결과 |
| 8/22 | Tuning | threshold/model 조정 |
| 8/23 | Final Experiment | benchmark + ablation |
| 8/24 | Productization | API/CLI/examples |
| 8/25 | OSS Quality | tests/docs/licenses |
| 8/26 | Submission Assets | report/video finalized |
| 8/27 | Reproduction only | clean machine 검증 + 제출 |

---

# 46. Milestones

## M0 — Infrastructure Ready

완료:

```text
Docker Compose
Qdrant
Neo4j
Python package
CI
```

---

## M1 — Baselines Ready

완료:

```text
Vector-only
Graph-only
Fixed Hybrid
```

이 milestone이 가장 중요하다.

여기까지 완료되면 Adaptive 기능이 일부 실패하더라도 제출 가능한 시스템이 남는다.

---

## M2 — Adaptive V0

완료:

```text
Query Analyzer
Rule Planner
Plan presets
```

---

## M3 — Adaptive V1

완료:

```text
Offline Profiler
Quality Model
Latency Model
Cost-aware Optimizer
```

---

## M4 — Infra Optimization

완료:

```text
Parallel Scheduler
Timeout
Fallback
Tracing
```

---

## M5 — Evidence Ready

완료:

```text
Benchmark
Ablation
Pareto graph
README
Demo
```

---

# 47. GitHub Epic / Issue Breakdown

## EPIC 1 — Infrastructure

```text
#1 Initialize Python package
#2 Add Docker Compose
#3 Configure Qdrant
#4 Configure Neo4j
#5 Add CI
#6 Add health endpoint
```

---

## EPIC 2 — Vector Retrieval

```text
#7 Define VectorBackend
#8 Implement Embedder
#9 Implement Qdrant ingestion
#10 Implement Qdrant search
#11 Add vector latency tracing
#12 Add vector integration tests
```

---

## EPIC 3 — Knowledge Graph

```text
#13 Define Entity/Relation models
#14 Implement entity extraction
#15 Implement relation extraction
#16 Implement Neo4j writer
#17 Implement entity resolver V0
#18 Add graph ingestion tests
```

---

## EPIC 4 — Graph Retrieval

```text
#19 Define GraphBackend
#20 Implement seed entity lookup
#21 Implement depth-1 traversal
#22 Implement depth-N traversal
#23 Recover associated chunks
#24 Add graph scoring
#25 Add graph tracing
```

---

## EPIC 5 — Hybrid

```text
#26 Define common RetrievalHit
#27 Implement deduplication
#28 Implement weighted RRF
#29 Implement fixed hybrid
#30 Add fusion tests
```

---

## EPIC 6 — Benchmark

```text
#31 Dataset loader
#32 Recall@K
#33 MRR
#34 nDCG
#35 Latency recorder
#36 Aggregate benchmark report
#37 Raw CSV export
```

---

## EPIC 7 — Query Planner

```text
#38 Define QueryFeatures
#39 Implement Analyzer
#40 Define RetrievalPlan
#41 Define PlanSpace
#42 Implement RulePlanner
#43 Add planner trace
```

---

## EPIC 8 — Cost Model

```text
#44 Build offline plan profiler
#45 Generate training matrix
#46 Train latency model
#47 Train quality model
#48 Model serialization
#49 Validation metrics
```

---

## EPIC 9 — Adaptive Optimizer

```text
#50 Score candidate plans
#51 Enforce latency constraint
#52 Select max-quality plan
#53 Implement optimizer fallback
#54 Explain selected plan
```

---

## EPIC 10 — Scheduler

```text
#55 Async vector execution
#56 Async graph execution
#57 Parallel hybrid execution
#58 Budget timer
#59 Branch cancellation
#60 Graceful degradation
```

---

## EPIC 11 — Productization

```text
#61 POST /v1/search
#62 CLI search
#63 CLI benchmark
#64 Example scripts
#65 Docker image
#66 Clean install test
```

---

## EPIC 12 — Competition

```text
#67 Final benchmark
#68 Ablation study
#69 Pareto charts
#70 README architecture
#71 License audit
#72 Development report
#73 Demo script
#74 3-minute video
#75 Clean-machine reproduction
```

---

# 48. PR Rules

PR 하나는 가능하면 하나의 명확한 목적만 가진다.

예:

```text
feat(vector): implement qdrant backend
```

좋음.

반면:

```text
feat: add qdrant, graph, planner, benchmark
```

피한다.

---

## PR Template

```markdown
## What

구현 내용

## Why

필요한 이유

## Changes

- ...
- ...

## Test

- [ ] Unit test
- [ ] Integration test
- [ ] Manual test

## Benchmark Impact

변화 없음 / 측정 결과

## Risks

...

## Checklist

- [ ] No secrets
- [ ] Docs updated
- [ ] Tests passed
```

---

# 49. Definition of Done for Each Feature

기능은 다음을 모두 만족해야 완료로 본다.

```text
Code implemented

Unit/integration test 존재

Error handling 존재

Trace 또는 metric 포함

README/example 필요 시 업데이트

No secret committed

CI pass
```

---

# 50. MVP Cut Line

마감 압박 시 반드시 살리는 기능:

## P0 — 절대 유지

```text
Qdrant
Neo4j

Vector baseline
Graph baseline
Fixed Hybrid

Benchmark harness

Query Analyzer

Rule Planner

Cost-aware Planner

Latency budget

Parallel execution

Fallback

CLI/API

Docker

README
```

---

## P1 — 시간이 있으면

```text
Reranker
Synthetic benchmark
Prometheus metrics
Rust fusion
```

---

## P2 — 대회 이후

```text
LanceDB adapter
NetworkX adapter
Milvus adapter

LangChain integration
LlamaIndex integration

Dashboard
Distributed execution
Online learning
```

---

# 51. Risk Register

## R1. Graph accuracy가 생각보다 낮음

원인:

```text
entity extraction 품질
relation extraction 품질
entity resolution 오류
```

대응:

1. controlled subset 사용
2. extraction 결과 cache
3. synthetic benchmark 병행
4. Graph schema 단순화

---

## R2. Adaptive가 Fixed Hybrid보다 성능이 안 나옴

대응:

```text
Rule Planner를 유지
Plan Space 축소
Query type별 threshold tuning
Cost model보다 offline lookup/table 사용 고려
```

중요:

Adaptive의 목적은 무조건 모든 metric을 이기는 것이 아니라:

```text
quality-latency tradeoff 개선
```

이다.

---

## R3. Graph latency가 지나치게 큼

대응:

```text
depth limit
seed limit
top_k limit
index 추가
parallel execution
timeout
fallback
```

---

## R4. Planner latency가 큼

대응:

```text
LLM router 제거
feature extraction 단순화
model 경량화
embedding 재사용
```

---

## R5. 시간이 부족함

즉시 제거:

```text
Dashboard
Rust
Extra adapters
Advanced reranker
```

---

# 52. Competition Demo Plan

## 0:00–0:20

문제 설명.

```text
Vector RAG:
semantic retrieval에 강함

Graph Retrieval:
relationship/multi-hop에 강함

하지만 모든 query에 Graph를 쓰면 latency 증가
```

---

## 0:20–0:40

제품 소개.

```text
Adaptive RAG Query Optimizer
```

Architecture 표시.

---

## 0:40–1:10

Simple query.

```text
"What is X?"
```

화면:

```text
Plan = VECTOR_FAST

Graph = OFF
```

---

## 1:10–1:45

Multi-hop query.

```text
"Where did the founder of X study?"
```

화면:

```text
relation = high

multi-hop = high

Plan = HYBRID_GRAPH_HEAVY

Graph depth = 2
```

Graph path 표시.

---

## 1:45–2:10

Latency budget 변경.

```text
300ms
→ 100ms
```

Plan 변경 표시.

---

## 2:10–2:45

Benchmark.

```text
Vector
Graph
Fixed Hybrid
Adaptive
```

Recall@10 + p95 latency.

---

## 2:45–3:00

마무리.

```text
RAG retrieval을 고정 방식이 아니라
query execution planning 문제로 다룬다.
```

---

# 53. Report Structure

개발보고서는 다음 흐름을 추천한다.

## 1. 문제 정의

Vector RAG / Graph Retrieval tradeoff.

## 2. 기존 접근

Vector-only / Graph-only / Fixed Hybrid.

## 3. 제안 방법

Adaptive Query Optimizer.

## 4. Architecture

Analyzer / Cost Model / Scheduler.

## 5. Implementation

Qdrant / Neo4j / Python.

## 6. Benchmark Methodology

Dataset / Metrics / Environment.

## 7. Results

Recall / Latency / Pareto.

## 8. Ablation

각 component 효과.

## 9. Open Source Design

Adapter / API / Docker / License.

## 10. Limitations

entity extraction, graph construction cost 등.

## 11. Future Work

online cost learning, additional DB adapters 등.

---

# 54. README First Screen

추천 구조:

```markdown
# AdaptiveRAG

Latency-aware query optimizer for Vector + Graph retrieval.

## Why?

Vector search is fast.
Graph retrieval understands relations.
Not every query needs both.

AdaptiveRAG selects a retrieval execution plan
based on query characteristics and latency budget.

## Quick Start

docker compose up -d

pip install -e .

adaptive-rag search "..."
```

그 아래 바로:

```text
Architecture
Benchmark
Quickstart
```

순서.

---

# 55. Final Success Criteria

## 기능

- [ ] Vector-only 검색 가능
- [ ] Graph-only 검색 가능
- [ ] Fixed Hybrid 가능
- [ ] Adaptive 검색 가능
- [ ] Latency budget 입력 가능
- [ ] Dynamic graph depth 가능
- [ ] Parallel execution 가능
- [ ] Timeout fallback 가능

## Benchmark

- [ ] Recall@10
- [ ] MRR@10
- [ ] nDCG@10
- [ ] p50
- [ ] p95
- [ ] p99
- [ ] Query type별 분석
- [ ] Latency budget별 분석
- [ ] Ablation

## OSS

- [ ] LICENSE
- [ ] THIRD_PARTY_LICENSES
- [ ] CONTRIBUTING
- [ ] Docker Compose
- [ ] Tests
- [ ] CI
- [ ] Examples
- [ ] Reproduction guide

## Competition

- [ ] 개발보고서
- [ ] 소스코드
- [ ] 3분 이내 시연영상
- [ ] clean-machine test
- [ ] benchmark raw result 보존
- [ ] 최종 결과 그래프

---

# 56. 최우선 실행 순서

프로젝트를 실제 개발할 때는 아래 순서를 절대 우선한다.

```text
1. Infrastructure

2. Vector-only

3. Graph-only

4. Fixed Hybrid

5. Benchmark

6. Query Analyzer

7. Rule Adaptive

8. Offline Plan Profiler

9. Cost Models

10. Cost-aware Optimizer

11. Parallel Scheduler

12. Timeout / Fallback

13. Final Benchmark

14. Documentation

15. Demo
```

잘못된 순서:

```text
Dashboard
→ Rust
→ LangChain
→ 추가 DB
→ Benchmark
```

이 순서는 피한다.

---

# 57. 가장 중요한 기술적 판단

프로젝트의 중심은 다음이 아니다.

```text
Qdrant
Neo4j
LangChain
GraphRAG
```

이들은 component다.

프로젝트가 직접 만들어야 할 핵심 자산은 다음 네 가지다.

```text
1. Query Analyzer

2. RetrievalPlan abstraction

3. Cost-aware Query Optimizer

4. Budget-aware Execution Scheduler
```

그리고 이 네 가지의 효과를:

```text
benchmark
```

로 증명해야 한다.

---

# 58. MVP Exit Criteria

아래 조건이 모두 충족되면 기능 개발을 중단하고 제출 준비 단계로 진입한다.

```text
Vector / Graph / Fixed / Adaptive
네 전략 모두 end-to-end 동작.

최소 500개 이상의 benchmark query를
자동 실행 가능.

동일 dataset과 환경에서
각 전략의 quality와 latency 측정 가능.

Latency budget 변경 시
실행계획 변경 확인 가능.

선택된 plan과 실행 trace 확인 가능.

Docker 기반 clean environment에서
설치 및 실행 가능.

README만 보고 신규 개발자가
example query를 실행 가능.
```

---

# 59. 제출 직전 금지 사항

마감 48시간 전부터:

```text
새 DB 추가 금지

대규모 refactoring 금지

Embedding model 교체 금지

Graph schema 변경 금지

새 framework 도입 금지

Rust rewrite 금지
```

허용:

```text
bug fix

test

benchmark 재실행

문서

영상

license

reproduction
```

---

# 60. Verified Technical Context / References

아래 자료는 본 PRD 작성 시 현재 기술 범위를 검증하기 위해 확인한 공식/원출처 중심 참고자료다.

## 2026 오픈소스 개발자대회

- 대회 공식 개요: https://osscontest.kr/overview
- NIPA 2026년 오픈소스 개발자대회 공고:
  https://www.nipa.kr/home/2-2/16815

확인한 핵심 사항:

```text
출품작 제출: 2026-08-27
제출물: 개발보고서, 소스코드/산출물, 3분 이내 시연영상
이후 기능 테스트 및 라이선스 검증 단계 존재
```

정확한 세부 규정은 제출 직전 공식 홈페이지 최신 공지를 다시 확인한다.

---

## Qdrant

Hybrid Query / Fusion 공식 문서:

https://qdrant.tech/documentation/search/hybrid-queries/

Qdrant는 공식 Query API에서 여러 retrieval 결과의 fusion과 RRF 계열 기능을 제공한다.

따라서 본 프로젝트의 차별점은 Qdrant 내부 hybrid retrieval 자체가 아니라:

```text
Qdrant Vector Result
+
Neo4j Graph Result
+
Query-aware Execution Planning
```

을 middleware 수준에서 최적화하는 데 둔다.

---

## Neo4j GraphRAG

공식 문서:

https://neo4j.com/docs/neo4j-graphrag-python/current/

Qdrant + Neo4j external retriever 문서:

https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html

현재 Neo4j GraphRAG Python package는 Qdrant 등의 외부 vector database와 Neo4j graph를 연결하는 retriever를 제공한다.

따라서 단순히:

```text
Qdrant + Neo4j 연결
```

하는 것만으로는 충분한 차별점이라고 보기 어렵다.

---

## Microsoft GraphRAG

공식 문서:

https://microsoft.github.io/graphrag/

GraphRAG는 Local Search, Global Search, DRIFT Search 등의 검색 방식을 제공한다.

본 프로젝트는 특정 GraphRAG 전략 자체보다 한 단계 위에서:

```text
Query 특징
+
Latency budget
+
Cost estimation
```

을 기반으로 retrieval execution plan을 선택하는 optimizer layer를 핵심으로 한다.

---

# 61. 최종 Product Statement

> **AdaptiveRAG는 자연어 질의를 단순히 검색하는 엔진이 아니라, 질의를 분석해 어떤 retrieval execution plan이 가장 적합한지를 결정하는 RAG Query Optimizer다. Vector Search의 속도와 Knowledge Graph의 관계 추론 능력을 고정적으로 결합하지 않고, query complexity와 latency budget에 따라 검색 범위·graph depth·top-k·fusion weight·reranking 여부를 동적으로 선택한다.**

본 프로젝트의 성공 여부는 기능 개수로 판단하지 않는다.

최종 판단 기준은 다음이다.

```text
"같은 resource budget에서 더 좋은 retrieval quality를 제공하는가?"

또는

"같은 retrieval quality를 더 낮은 latency로 제공하는가?"
```

이를 reproducible benchmark로 증명하는 것이 최종 목표다.
