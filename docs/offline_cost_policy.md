# Stage 12 Research-only Offline Cost Policy

Stage 12 evaluates the frozen Stage 11 R2 models on the validation split only. It is deliberately
isolated from `BaselineSearchEngine`, REST, and the CLI search path. Public `planner=cost_aware`
therefore continues to return `MODE_UNAVAILABLE`; the default planner remains Rule.

## Why this path is offline-only

The R2 model bundle failed three acceptance gates and is marked `research_only`:

- plan-pair ranking accuracy is below 0.70;
- P0/P1 have no valid latency labels, so per-plan latency coverage fails;
- pinball-loss improvement is below 0.10.

Its frozen feature schema also includes benchmark source and query-tag one-hot columns. Those
columns are known in the benchmark matrix but are not available in a normal online request.
`OfflineResearchContext` makes that dependency explicit rather than inventing online values or
claiming runtime compatibility.

## Selection contract

For each validation query and each 50/100/200/500 ms budget, the optimizer:

1. encodes all enabled P0 plans (`P0,P1,P2,P3,P4,P5,P6,P8`);
2. records predicted Recall@10, predicted conditional p95 latency, model version, and input hash;
3. rejects non-finite or out-of-range predictions instead of silently using them;
4. accepts a candidate only when `predicted_p95 + finalization_reserve <= remaining_budget`;
5. selects maximum predicted Recall@10, then lower predicted latency, lower graph depth, and lower
   plan ID;
6. selects P0 best effort with `budget_feasible=false` when no candidate is feasible.

The evaluator computes Rule and cost-aware decisions side by side but performs no retrieval. It
joins the selected decision to already-frozen Stage 9/10 measurements. Rule and BestFixed metrics
come from Stage 9 measured validation trials; cost-aware and Oracle evidence comes from the Stage
10 validation matrix. The held-out test split is never read.

## Run

```bash
uv run python scripts/evaluate_cost_policy.py
```

The command checksum-verifies and safely loads the `.skops` artifacts, then writes:

- `artifacts/cost_models/stage12_r2/decisions.jsonl` — 480 redacted decisions with all 3,840
  candidate estimates;
- `artifacts/cost_models/stage12_r2/policy_report.json` — aggregate comparisons, disagreement,
  planner overhead, and guard simulation;
- `benchmark/manifests/stage12_policy_evidence_r2.json` — small checksum-bound evidence manifest.

## Frozen R2 result

| Metric | Result |
|---|---:|
| Validation query-budget groups | 480 |
| Candidate estimates | 3,840 |
| Cost-aware mean Recall@10 | 0.005507 |
| Recall difference vs Rule | +0.005507 |
| Recall difference vs BestFixed | +0.000726 |
| Cost-aware budget-violation rate | 0.005000 |
| Violation difference vs Rule | +0.004792 |
| Violation difference vs BestFixed | -0.073958 |
| Oracle-comparable groups | 382 |
| Mean regret vs Oracle | 0.002618 |
| Cost-aware/Rule decision disagreement | 427/480 (0.889583) |
| Cost-aware planner overhead p95 | ~10 ms (host-dependent) |
| Rejected candidate predictions | 1,856 |
| No predicted-feasible groups | 120 |

These are research measurements, not a serving claim. In particular, the high invalid-prediction
count reflects finite predictions outside the frozen model's valid output range; fail-closed
candidate filtering exposes rather than hides this extrapolation.

## Calibration guard simulation

The guard uses a rolling 100-observation window and begins evaluating after 20 observations. It
latches the artifact off when budget violations exceed 0.10 or actual execution latency exceeds
predicted p95 more than 0.20 of the time. A latch cannot clear itself; only explicit loading of a
different compatible artifact resets it.

On R2 Stage 10 measured execution trials, the guard disabled the bundle after 319 evaluable
observations because the rolling underprediction rate reached 0.21. The remaining 422
query-budget groups would route to Rule. This outcome, the inherited Stage 11 gate failures, and
the offline-only feature metadata jointly prohibit public activation.
