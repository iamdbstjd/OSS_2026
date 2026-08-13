# Graph extraction audit v1 / 그래프 추출 감사 v1

This directory is a frozen human-review work queue bound to the Stage 2 train split.
이 디렉터리는 Stage 2 train split에 결박된 사람 검토 작업 큐입니다.

- `sample_v1.jsonl`: 100 SHA-256-ordered train sentences and system predictions.
- `manifest_v1.json`: corpus, benchmark, split, extractor, sample IDs, and checksums.
- `reviews_v1.jsonl`: incomplete review template. Never treat `completed:false` as gold.
- `evaluation_v1.json`: current audit result.
- `configs/graph_tier_policy.json`: fail-closed rule-planner handoff.

Review all 100 primary rows. Independently review the 20 rows marked `secondary`, then
fill the matching `adjudicator` rows. A completed row must have a non-placeholder
`reviewer_id`, JSON arrays for `entities` and `relations`, and `completed:true`. Validate
the filled records as `AuditReview`, then call `evaluate_graph_audit`. Until all required
reviews exist, the rule graph tier remains disabled; graph mode may only be used as an
explicit comparison baseline.

```bash
uv run python scripts/evaluate_graph_audit.py
```

100개 primary row를 모두 검토하고, `secondary`로 표시된 20개는 두 번째 검토자가
독립적으로 검토한 뒤 대응하는 `adjudicator` row를 확정해야 합니다. 완료 row에는 실제
reviewer ID, entity/relation JSON 배열, `completed:true`가 필요합니다. 모든 필수 검토가
끝나기 전에는 rule graph tier가 비활성화되며 graph mode는 명시적 비교 baseline으로만
사용할 수 있습니다.

The source passages retain the dataset licenses and attribution documented in
`benchmark/manifests/licenses.yaml` and `benchmark/NOTICE.md`.
