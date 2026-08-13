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
| spaCy en_core_web_sm | `3.8.0` | Wheel `sha256:1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85` | MIT | https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0 | Installed from the exact hashed wheel in `uv.lock`; used only for deterministic offline entity/relation extraction. |

## Datasets

| Dataset | Upstream version/date | Official URL | Download URL | License / terms | Redistribution | Required attribution | Raw archive SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Natural Questions / DPR | DPR artifact, last modified 2020-03-18 | https://ai.google.com/research/NaturalQuestions/download | https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz | CC BY-SA 3.0 | Raw archive is downloaded locally and not redistributed | Natural Questions authors and DPR preprocessing authors | `3249231587e8140e3794c060b0233afc61f4fa5e40b6a172d59519af5fe40c73` |
| HotpotQA distractor train | v1.1; repository commit `3635853403a8735609ee997664e1528f4480762a` | https://hotpotqa.github.io/ | Official Hugging Face organization mirror, revision `1908d6afbbead072334abe2965f91bd2709910ab` (two shards; see manifest) | CC BY-SA 4.0 | Raw shards are downloaded locally and not redistributed | Yang et al., *HotpotQA*, EMNLP 2018 | `76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68`, `713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f` |
| MuSiQue-Ans train | v1.0; repository commit `922ac98f19a201998dbdae6d7f2887a5258dbdeb` | https://github.com/StonyBrookNLP/musique | Upstream Google Drive archive (ID recorded in `benchmark/manifests/licenses.yaml`) | CC BY 4.0 | Raw archive is downloaded locally and not redistributed | Trivedi et al., *MuSiQue*, TACL 2022 | `98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd` |

The machine-readable audit, including every official/actual URL, byte size,
license URL, and redistribution decision, is
`benchmark/manifests/licenses.yaml`. The derived benchmark query manifest and
qrels retain source attribution; the multi-gigabyte raw inputs and normalized
corpus text remain ignored local cache artifacts.

## Python dependencies

`uv.lock` is the authoritative, complete dependency inventory. At release time,
export or attach the resolved package names, versions, licenses, and source
metadata produced from that lockfile.

| Package | Resolved version | License / terms | Source | Included by |
| --- | --- | --- | --- | --- |
| _Populate from `uv.lock` at release_ | _TBD_ | _TBD_ | PyPI / upstream | runtime, development, benchmark, or graph-extraction |
