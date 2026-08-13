# Frozen metric contract

The executable, dependency-free implementations are in
`src/ragplan/benchmark/metrics.py`. They define:

- Recall@5 and Recall@10 with `relevance_grade >= 1` as relevant;
- MRR@10 with `relevance_grade >= 1` as relevant;
- nDCG@10 with grades 0/1/2 and gain `2^grade - 1`.

Hand-calculated reference inputs and expected values are frozen in
`tests/fixtures/benchmark/metric_cases.json` and checked by
`tests/benchmark/test_metrics.py`. Duplicate retrieved chunk IDs are counted at
their first occurrence only, and a query with zero relevant chunks is rejected
instead of silently contributing zero.
