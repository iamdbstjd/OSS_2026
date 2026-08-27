# Runtime Operations

This document contains executable operational examples for the implemented retrieval runtime.
Design drafts and internal planning documents are intentionally not part of the public repository.

## Explicit graph retrieval

Graph retrieval requires an activated corpus, the matching graph-stage manifest, the pinned
extractor lockfile, and a Neo4j password supplied only through the environment.

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

Traversal uses exact-normalized seeds, one to three `RELATES_TO` hops, direction-preserving
provenance, and bounded recovery through `MENTIONS`.

## Explicit fixed hybrid retrieval

The fixed-hybrid path verifies the active pointer and both stage manifests before starting. Vector
and Graph branches execute in parallel and merge through `weighted_rrf_v1`.

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

## Scheduler contract

`runtime_semantics_version=v1` freezes the shared execution boundary used by the API, CLI,
benchmark, and profiler. Requests move through
`received → analyzing → planning → executing → fusing → complete|partial` under one monotonic
absolute deadline. The engine records branch start and finish boundaries, remaining budget,
cancellation reason, timeout origin, and circuit state.

The scheduler provides:

- at most 32 in-flight requests with queue-free admission;
- parallel execution of at most two retrieval branches;
- finalization reserve of `min(20ms, max(5ms, budget × 5%))`;
- per-backend five-failure/30-second circuits with one half-open probe;
- typed partial responses that preserve a successful sibling branch;
- full child-task cleanup after client disconnect.

## Rule query analysis

The Rule path uses the frozen Stage 8 `qf_v1` query-feature contract. It computes language support,
token and entity counts, relation, multi-hop, comparison, aggregation, and global signals once per
request. Threshold provenance is validation-only, and held-out test IDs are not accepted as tuning
input.

The planner combines these features with remaining budget, static p95 profiles, graph-audit state,
and circuit state from `configs/rule_planner_v1.json`. When graph capability is unavailable, the
Rule path remains vector-only. Its trace records the selected plan, matched rules, feasibility estimates,
fallback reason, and feature/config versions without storing the raw query or embedding.

## Chunker evidence versions

- `token-window-220-overlap-40-v1` reconstructs text by decoding token IDs and remains the immutable
  contract for the frozen benchmark and existing corpus evidence.
- `token-window-220-overlap-40-v2` reconstructs text from fast-tokenizer source offsets and preserves
  source case for new Graph NER corpora and sample-v2.

Vector-stage manifests record the selected version. Corpus activation requires the declared source
version to match that manifest exactly, preventing v2 content from being presented as v1 evidence.
