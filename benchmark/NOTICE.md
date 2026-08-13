# Benchmark data notice

The RAGPlan source code is Apache-2.0, but the derived benchmark artifacts in
this directory contain identifiers, questions, answers, and annotations from
upstream datasets under their own terms:

- Natural Questions / DPR-derived records: CC BY-SA 3.0. Attribution: the
  Natural Questions authors and the DPR preprocessing authors.
- HotpotQA v1.1-derived records: CC BY-SA 4.0. Attribution: Yang et al.,
  *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*,
  EMNLP 2018.
- MuSiQue v1.0-derived records: CC BY 4.0. Attribution: Trivedi et al.,
  *MuSiQue: Multihop Questions via Single-hop Question Composition*, TACL 2022.

See `manifests/licenses.yaml` for exact source and download URLs, immutable raw
checksums, upstream revisions, license links, and redistribution decisions.
Raw source archives and normalized corpus text are intentionally excluded from
Git and Docker images. The primary 600-query benchmark must not be combined
with the separately identified synthetic graph fixture when reporting primary
results.
