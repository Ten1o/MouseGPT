# Example workflows

Explore MouseGPT with a small set of vision-language inputs or a prepared behavioral session. Both bundles are hosted in [MouseGPT-examples on Hugging Face](https://huggingface.co/datasets/Ten1o/MouseGPT-examples); the downloader installs them into the directory structure used by the code.

| Example | Download | Contents | Start here |
| --- | ---: | --- | --- |
| `vlm-demo` | 0.29 MB | 10 images with prompts and example responses | [Vision-language inference](../2_VLM/readme.md) |
| `nlui-demo` | 985.32 MB | One processed session and 6,000 frame previews | [Natural-language interface](../4_NLUI/readme.md) |

Download sizes use decimal MB. The NLUI archive expands to approximately 1.39 GB.

## 1. Inspect vision-language inputs

Use Python 3.10 or newer. From the repository root, download and validate the small example:

```bash
python -m pip install -r requirements-assets.txt
python scripts/download_assets.py vlm-demo
python 2_VLM/prompt_test/client_test.py --validate-only
```

This creates `2_VLM/prompt_test/data/test_data.jsonl` and an `imgs/` directory containing 10 JPEGs. Each record has three fields:

| Field | Description |
| --- | --- |
| `query` | The prompt sent to the vision-language model. |
| `images` | Image paths relative to the JSONL file. |
| `response` | An example model response, included for illustration. |

Validation runs locally without a model server. To generate new descriptions with a hosted third-party VLM, follow the [researcher quick start](../README.md#quick-start-use-a-hosted-vlm): install `requirements-api.txt`, configure the VISION and EMBEDDING roles, then run:

```bash
python 2_VLM/prompt_test/client_test.py \
  --backend api --limit 3 --output-dir runs/api-demo/demo
```

This produces validated descriptions and matching frame previews for downstream embedding, packaging, and search. It requires no local MouseGPT checkpoint or GPU. The client sends only prompts and images; stored responses are neither model input nor expert reference annotations. To use local MouseGPT weights instead, follow [SWIFT model serving](../2_VLM/readme.md#serve-a-mousegpt-model) and check the [model repository](https://huggingface.co/Ten1o/MouseGPT) for checkpoint availability.

## 2. Explore a prepared session

```bash
python scripts/download_assets.py nlui-demo
python scripts/download_assets.py nlui-demo --verify
```

The installed files are:

```text
4_NLUI/data/
├── 240419_C11M4.h5
└── 240419_C11M4_imgs/
    └── *.jpg
```

The HDF5 contains descriptions and cached embeddings for 5,999 frames. Every stored frame has an image; the additional preview `00037368.jpg` has no cached feature row. Align images and records by **frame ID**, rather than image-list position.

Follow the [Stage 4 guide](../4_NLUI/readme.md) to inspect descriptions, retrieve frames, search behavior, and use the available analysis tools. Pose estimation and MouseGPT inference do not need to be rerun for this prepared session. Conversation, semantic search, and some analysis tools require the external APIs described in that guide. The published session uses official OpenAI `text-embedding-3-large` vectors; its exact file hash is recognized for compatibility. Changing the embedding service/model requires regenerating the vectors and packaging a new session from the descriptions.

## Offline installation and integrity checks

For offline installation, place the published `vlm-demo.zip` and `nlui-demo.zip` files in a local directory:

```bash
python scripts/download_assets.py all --local-bundle-dir /path/to/bundles
python scripts/download_assets.py all --verify
```

Online downloads use the dataset revision pinned in [assets/manifest.json](../assets/manifest.json). Both installation methods verify archive SHA-256 hashes and file counts. `--verify` checks the installed files without downloading them again.

The downloader preserves existing data. To install into a separate directory layout, add `--root /path/to/another/MouseGPT`.

## Data scope

These examples support individual workflow components. They contain processed data and inference inputs; raw videos, camera calibration, model checkpoints, training splits, the full experimental cohort, and the 435-frame benchmark are separate assets. A single prepared session cannot reproduce cohort-level phenotype or group statistics. Full raw data are available upon request; see the repository's [data availability statement](../README.md#data-availability).
