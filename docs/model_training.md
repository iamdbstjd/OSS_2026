# Stage 11 Cost Model Training

Stage 11 consumes the checksum-bound Stage 10 training matrix. It never reads the held-out test
split and never uses raw query embeddings.

## Feature schema

`cost_model_features_v1:qf_v1` contains:

- nine qf_v1 numeric query features;
- nine immutable plan numeric/boolean features;
- fixed one-hot categories for dataset source, query tags, and P0 plan ID.

Feature ordering is frozen in `ragplan.planner.training.FEATURE_NAMES`. One query ID must belong to
exactly one split, query features must remain identical across plans/budgets, and plan features must
remain identical for one plan ID. Duplicate query-plan-budget rows, test rows, mixed identity
bundles, missing values, NaN, and infinity are rejected.

Quality training uses every row with ten valid quality trials. Latency training uses only rows with
all ten exact scheduler-to-result execution labels. The R2 matrix provides 11,520/3,840
train/validation quality rows and 5,976/2,108 latency rows. P0 and P1 contain no complete latency
labels, so their per-plan coverage is explicitly zero rather than silently omitted.

## Models and metrics

- Quality: deterministic `HistGradientBoostingRegressor`, squared-error target Recall@10, predictions
  clipped to `[0, 1]`.
- Latency: deterministic quantile `HistGradientBoostingRegressor`, quantile 0.95, target conditional
  p95 execution latency.
- Ranking accuracy excludes exact target ties and compares plans only within the same query-budget.
- Pinball improvement compares against train-derived constant per-plan p95 predictions.
- Policy simulation uses `predicted_p95 + finalization_reserve <= budget`, falls back to P0 when no
  plan is predicted feasible, and compares with measured Rule and Oracle evidence.

## Artifact safety

Only `.skops` is supported. Loading performs, in order:

1. manifest validation;
2. artifact SHA-256 verification;
3. repository-fixed trusted-type inspection;
4. feature/catalog/corpus/qrels/model/backend/dependency/runtime compatibility checks;
5. hardware mismatch warning;
6. serving-eligibility gate.

`pickle`, `joblib`, `cloudpickle`, unknown estimator types, unknown skops types, checksum corruption,
and critical compatibility drift are rejected. A hardware-only mismatch permits inspection but
blocks default activation.

## R2 validation outcome

The frozen R2 run is `research_only`.

| Metric | Result | Gate |
|---|---:|---:|
| Quality MAE | 0.009219 | <= 0.10 PASS |
| Plan-pair ranking accuracy | 0.681529 | >= 0.70 FAIL |
| Quality predicted-best regret | 0.003103 | <= 0.05 PASS |
| Latency coverage overall | 0.928368 | >= 0.90 PASS |
| Minimum per-plan coverage | 0.0 (P0/P1 missing) | >= 0.85 FAIL |
| Severe underprediction | 0.018975 | <= 0.02 PASS |
| Pinball improvement | 0.047897 | >= 0.10 FAIL |
| Policy Recall difference vs Rule | +0.007590 | >= -0.01 PASS |
| Policy violation-rate difference | +0.006250 | <= 0.02 PASS |
| Policy regret vs Oracle | 0.0 | <= 0.05 PASS |

Because any failed gate makes the complete bundle research-only, neither artifact may be loaded for
serving. Stage 12 may use the bundle only for offline/research comparison while public behavior
continues to fall back to Rule.
