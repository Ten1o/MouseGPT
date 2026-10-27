# MouseGPT

**Vision-language models and natural-language tools for mouse behavior analysis**

[![Paper — arXiv](https://img.shields.io/badge/arXiv-2503.10212-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.10212)
[![GitHub — Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/Ten1o/MouseGPT)
[![Hugging Face — Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/Ten1o/MouseGPT)
[![Hugging Face — Example data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Example%20data-FFD21E)](https://huggingface.co/datasets/Ten1o/MouseGPT-examples)

MouseGPT combines visual observations with pose-derived kinematic information to describe mouse posture and movement in natural language. These descriptions provide semantic representations for behavior retrieval, classification, clustering, and exploration through a conversational interface.

This repository accompanies **[MouseGPT: A Large-scale Vision-Language Model for Mouse Behavior Analysis](https://arxiv.org/abs/2503.10212)** and contains the training, inference, preprocessing, and analysis code for all four stages of the framework.

> **Model release:** MouseGPT checkpoints will be uploaded to [Hugging Face](https://huggingface.co/Ten1o/MouseGPT). The code and example data are provided separately, so you can inspect model inputs and explore a prepared session while the checkpoint uploads are pending. See the [model release checklist](https://huggingface.co/Ten1o/MouseGPT#todo-model-release) for progress.

## Overview

MouseGPT connects video-based measurements to interpretable behavioral descriptions and downstream analysis. You can run the complete workflow with the required data and checkpoints, or begin with prepared inputs at a later stage.

| Stage | What it provides | Guide |
| --- | --- | --- |
| **1. Keypoints and kinematics** | Animal detection, 2D keypoint estimation, and multi-view 3D reconstruction | [Pose pipeline](1_keypoints/readme.md) |
| **2. Vision-language modeling** | Image and kinematic preprocessing, model training, deployment, and structured behavior descriptions | [VLM pipeline](2_VLM/readme.md) |
| **3. Behavior analysis** | Description embeddings, coarse and fine behavior categories, unusual-frame ranking, and cohort analysis | [Analysis workflows](3_applications/readme.md) |
| **4. Natural-language interface** | Conversational access to session descriptions, semantic search, classification, and clustering | [Run the interface](4_NLUI/readme.md) |

The [benchmark toolkit](benchmark/README.md) provides a local workbench for frame-description evaluation, versioned references and scoring, and [figure reproduction](benchmark/figure3/README.md).

## Getting started

For API-backed workflows, copy `.env.example` to `.env.local` at the
repository root and fill `MOUSEGPT_API_KEY`. The default service is the
official OpenAI API. [Configuration guide](CONFIGURATION.md) explains optional
Azure/compatible endpoints and separate settings for different stages.

### 1. Clone the repository

The example downloader and input validation require Python 3.10 or newer. They do not require a GPU.

```bash
git clone https://github.com/Ten1o/MouseGPT.git
cd MouseGPT
python -m pip install -r requirements-assets.txt
```

If you already have data, this read-only check suggests where to start and
whether that stage's API role is configured. It does not process files or make
an API request:

```bash
python scripts/check_workflow.py --input /path/to/session
```

### 2. Inspect the VLM examples

Download the small image-and-prompt bundle and validate its inputs:

```bash
python scripts/download_assets.py vlm-demo
python scripts/download_assets.py vlm-demo --verify
python 2_VLM/prompt_test/client_test.py --validate-only
```

The bundle contains **10 images and 10 prompt/response records** and downloads approximately **0.29 MB**. It shows the input format and representative descriptions; stored responses are illustrative model outputs. The validation command checks the records and image paths without contacting a model.

To generate new descriptions, follow the [Stage 2 guide](2_VLM/readme.md) to deploy a compatible checkpoint and run the inference client.

### 3. Explore a prepared session

For downstream analysis and the natural-language interface, download the session bundle:

```bash
python scripts/download_assets.py nlui-demo
python scripts/download_assets.py nlui-demo --verify
```

This download is approximately **985 MB** and expands to **1.39 GB**. It contains the `240419_C11M4` session with **5,999 stored frame records** and **6,000 JPEG previews**. Follow the [Stage 4 guide](4_NLUI/readme.md) to install the interface environment and configure its API access. The prepared descriptions and embeddings let you start here without running pose estimation or MouseGPT inference.

See the [examples guide](examples/README.md) for file formats, frame alignment, local archive installation, and choosing a workflow. Downloads use a pinned dataset revision and SHA-256 verification. For an access-controlled Hugging Face repository, authenticate with `hf auth login` or set `HF_TOKEN`.

## Installation and requirements

Each stage has its own dependencies. Use the environment described in that stage's guide instead of installing every component into a single environment.

| Workflow | Requirements |
| --- | --- |
| Download examples and validate VLM inputs | Python 3.10+; no model checkpoint or GPU |
| Pose estimation and MouseGPT inference | Compatible checkpoints, model configuration files, and a CUDA/PyTorch environment |
| Analyze prepared descriptions and embeddings | The Stage 3 scientific Python environment; API credentials for model-assisted operations |
| Run the natural-language interface | The Stage 4 environment, a paired HDF5/image session, and `.env.local` with `MOUSEGPT_API_KEY` |
| Use the benchmark workbench | Python 3.12+; figure rendering has additional dependencies listed in its guide |

The repository includes MouseGPT integrations with MMPose, MMYOLO, and SWIFT. Install the bundled sources where instructed so that the project-specific configurations and dataset registrations are available. Provider-assisted annotation and analysis send the selected inputs to the configured service and may incur API charges.

## Models

| Model | Foundation model | Role | Weights |
| --- | --- | --- | --- |
| **MouseGPT-Large** | InternVL2-Llama3-76B | Detailed vision-language behavior descriptions | Upload pending |
| **MouseGPT-Lite** | MiniCPM-V 2.6, 8B | A smaller vision-language model for behavior descriptions | Upload pending |
| **Kinematic estimation models** | Detector and keypoint models | Visual measurements for pose reconstruction and kinematic preprocessing | Upload pending |

Checkpoints and their loading instructions will be maintained in the [Hugging Face model repository](https://huggingface.co/Ten1o/MouseGPT). All model training, deployment, and inference code is maintained here on GitHub.

## Data availability

[MouseGPT-examples](https://huggingface.co/datasets/Ten1o/MouseGPT-examples) hosts the VLM examples and prepared analysis session. These bundles support learning the input formats and trying individual components. Training, reconstruction, and cohort-level analyses require the corresponding recordings, calibration, annotations, and experimental groups described in the module guides.

**Full raw data are available upon request.** Contact the corresponding authors listed in the [paper](https://arxiv.org/abs/2503.10212) with a description of the requested data and intended use.

## Citation

If MouseGPT supports your research, please cite the preprint:

```bibtex
@misc{xu2025mousegpt,
  title         = {MouseGPT: A Large-scale Vision-Language Model for Mouse Behavior Analysis},
  author        = {Teng Xu and Taotao Zhou and Youjia Wang and Peng Yang and
                   Simin Tang and Kuixiang Shao and Zifeng Tang and Yifei Liu and
                   Xinyuan Chen and Hongshuang Wang and Xiaohui Wang and
                   Huoqing Luo and Jingya Wang and Ji Hu and Jingyi Yu},
  year          = {2025},
  eprint        = {2503.10212},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2503.10212},
  url           = {https://arxiv.org/abs/2503.10212}
}
```

## License and acknowledgments

Original MouseGPT-authored material is distributed under [CC BY-NC-ND 4.0](LICENSE), subject to the component exclusions in that notice. The [MMYOLO detector components and copied configurations](LICENSES/GPL-SCOPE.md), including MouseGPT modifications to them, are distributed under GPLv3 without the CC noncommercial or no-derivatives restrictions. MMPose- and SWIFT-derived source retains its Apache-2.0 terms and notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and integration details. Datasets and model checkpoints have separate asset-specific terms.

We thank the developers of MMPose, MMYOLO, SWIFT, EasyMocap, and the foundation models used by MouseGPT.

## Support

For installation questions, reproducible bugs, or documentation improvements, open a [GitHub issue](https://github.com/Ten1o/MouseGPT/issues). Include the relevant stage, source commit, operating system, dependency versions, and a minimal example where possible. For data requests, use the correspondence information in the paper.
