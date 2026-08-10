# Contributing to RAGPlan

Thanks for contributing. RAGPlan is a Python 3.12 project managed with
[uv](https://docs.astral.sh/uv/).

## Development setup

1. Fork and clone the repository.
2. Copy `.env.example` to `.env` only if local services need configuration.
   Never commit `.env`, credentials, access tokens, model weights, or raw data.
3. Install the locked environment:

   ```bash
   uv sync --frozen
   ```

   Install an optional group only when working on that area, for example
   `uv sync --frozen --group benchmark`.

4. Run the focused checks before opening a pull request:

   ```bash
   uv run ruff format --check .
   uv run ruff check .
   uv run mypy src
   uv run pytest -m "not integration and not e2e"
   uv run python -c "import ragplan"
   ```

Use `uv lock` intentionally when a dependency changes, and commit the matching
`pyproject.toml` and `uv.lock` updates together. Direct URL dependencies must be
pinned and require an explicit project decision.

## Pull requests

- Keep each pull request focused on one stage task or one cross-cutting contract.
- Add or update tests for changed behavior; do not weaken assertions to hide a
  regression.
- Run the checks above and describe any validation that cannot run locally.
- Update public documentation, configuration samples, and attribution records
  when a user-visible behavior or third-party artifact changes.
- Do not start offline profiling or publish cost-aware latency claims before the
  scheduler/deadline semantics are frozen.

## Contracts, privacy, and data

Public API, CLI, benchmark, and profiler paths must use `RAGPlanEngine.search()`;
do not create benchmark-only retrieval implementations. IDs, plans, traces,
datasets, models, and configuration artifacts must be versioned or hashed.

Default traces are redacted: do not log raw queries, embeddings, full document
text, credentials, or API exception stack traces. Raw query recording is allowed
only for the frozen public benchmark with explicit `logging.mode=benchmark`.

Do not commit raw benchmark datasets, model weights, large benchmark results, or
model artifacts. Record each dataset/model download's provenance, license, and
checksum in the relevant manifest and `THIRD_PARTY_LICENSES.md`.

## Code style and review

Follow the existing Ruff and mypy configuration. Prefer small, typed, testable
changes over new frameworks or adapters. Retrieval, database, LLM-routing, and
reranking dependencies require explicit scope approval in P0.

By submitting a contribution, you agree to license it under Apache-2.0.
