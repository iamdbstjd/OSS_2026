# Security Policy

## Supported versions

Security fixes are applied to the current default branch during the P0 project
period. RAGPlan is a local reproducible MVP, not a production-hosted service.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Report it
privately to the repository maintainers through the repository's private
security-advisory channel. Include a description, affected version or commit,
reproduction steps, impact, and any suggested mitigation.

Maintainers will acknowledge the report, assess scope, coordinate a fix, and
publish a disclosure when users can safely update. Do not include credentials,
private user queries, model weights, or non-public datasets in a report.

## Security and privacy baseline

- Keep API keys, database passwords, tokens, and `.env` files out of Git.
  `.env.example` may contain only clearly marked demo values.
- Do not log raw queries, query embeddings, full document text, credentials, or
  API exception stack traces in the default (`redacted`) trace mode.
- Raw query logging is permitted only for a frozen public benchmark when
  `logging.mode=benchmark` is explicitly set.
- Parameterize Cypher queries; never interpolate user-provided text into Cypher.
- Do not package artifacts with unknown licenses into the repository or Docker
  images. Pin dependencies, container tags/digests, model revisions/checksums,
  and dataset provenance as documented in `THIRD_PARTY_LICENSES.md`.

## Scope and operational limits

The repository does not accept production secrets or live user data. If you
believe a public artifact contains sensitive data, stop redistributing it and
report it privately with the artifact path and commit identifier.
