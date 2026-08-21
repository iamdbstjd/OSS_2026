# Examples

`llm_handoff.py` demonstrates the intentional boundary between RAGPlan retrieval and answer
generation. It consumes a complete `SearchResponse`, keeps result ranking and chunk provenance,
and emits provider-neutral `system`/`user` messages. It never imports or calls an LLM SDK.

```bash
uv run ragplan search \
  --query "What did Ada Lovelace write about?" \
  --planner vector \
  --pretty > /tmp/ragplan-response.json

uv run python examples/llm_handoff.py \
  --response /tmp/ragplan-response.json \
  --question "What did Ada Lovelace write about?"
```

The generated prompt treats retrieved text as untrusted evidence rather than instructions. The
caller remains responsible for choosing an LLM, credentials, retention policy, and generation
evaluation.
