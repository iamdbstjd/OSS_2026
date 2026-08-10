# Third-Party Licenses and Reproducibility Record

RAGPlan is licensed under [Apache-2.0](./LICENSE). This file records the
third-party components used to reproduce a release. It is a release artifact:
every populated row must identify the exact version or revision and its source.

Do not add an artifact with an unclear license to the repository or a project
Docker image. Raw benchmark datasets and model weights are not redistributed by
this repository.

## Update rules

- Python packages are pinned in `uv.lock`; do not add unpinned direct-URL
  dependencies.
- Container rows require both an immutable image digest and the tested tag.
- Model rows require an upstream revision and SHA-256 for every downloaded file.
- Dataset rows require the official and actual download URLs, upstream version,
  license/terms, redistribution assessment, attribution, and raw archive
  SHA-256.
- Keep this file synchronized with `uv.lock`, Compose configuration, manifests,
  and release artifacts. A placeholder is not approval to redistribute.

## Qdrant

| Component | Tested tag | Image digest | License / terms | Source | Notes |
| --- | --- | --- | --- | --- | --- |
| Qdrant | `v1.18.2` | `sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c` | Apache-2.0 | https://github.com/qdrant/qdrant/releases/tag/v1.18.2 | OCI index digest; validate the selected platform during clean-machine smoke. |

## Neo4j

| Component | Tested tag | Image digest | License / terms | Source | Notes |
| --- | --- | --- | --- | --- | --- |
| Neo4j Community | `5.26.28` | `sha256:ff32db30b2baff97971e441b46bfd9c832c1b62c970398ef579244c06b21d357` | GPL-3.0-only | https://neo4j.com/release-notes/database/ | Community Edition OCI index digest; keep it a separate service and preserve GPL notice. |

## Models

| Model | Upstream revision | File SHA-256 | License / terms | Source | Redistribution / attribution |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | `b8903db39f65d93ae28d49a37c4f3fa90c5f94e0` | Manifest `f0635a959113bf3817d63ecd3a02b9c840795cbd17afeab9af596c9226771574`; all 10 downloaded-file SHA-256 values are recorded in `configs/models/all_minilm_l6_v2.b8903db.manifest.json` | Apache-2.0 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/b8903db39f65d93ae28d49a37c4f3fa90c5f94e0 | Weights are downloaded locally from the immutable revision and are not redistributed. |

## Datasets

| Dataset | Upstream version/date | Official URL | Download URL | License / terms | Redistribution | Required attribution | Raw archive SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Natural Questions / DPR | _TBD_ | https://ai.google.com/research/NaturalQuestions | _TBD_ | CC BY-SA 3.0 | Do not redistribute raw archive | Preserve source and license attribution | _TBD_ |
| HotpotQA distractor train | v1.1 | https://hotpotqa.github.io/ | _TBD_ | CC BY-SA 4.0 | Do not redistribute raw archive | Preserve source and license attribution | _TBD_ |
| MuSiQue-Ans train | v1.0 | https://github.com/StonyBrookNLP/musique | _TBD_ | CC BY 4.0 | Do not redistribute raw archive | Preserve source and license attribution | _TBD_ |

## Python dependencies

`uv.lock` is the authoritative, complete dependency inventory. At release time,
export or attach the resolved package names, versions, licenses, and source
metadata produced from that lockfile.

| Package | Resolved version | License / terms | Source | Included by |
| --- | --- | --- | --- | --- |
| _Populate from `uv.lock` at release_ | _TBD_ | _TBD_ | PyPI / upstream | runtime, development, benchmark, or graph-extraction |
