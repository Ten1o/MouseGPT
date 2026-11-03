# MouseGPT

**From mouse images to behavioral descriptions, search, and analysis**

**Browser UI:** See the [UI user guide](ui/README.md) for clone-to-launch instructions, environment/API setup, stage inputs and outputs, and current implementation/validation status. Python 3.10+ is required to start the UI; Node.js/npm is not required.

[![Paper — arXiv](https://img.shields.io/badge/arXiv-2503.10212-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.10212)
[![GitHub — Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/Ten1o/MouseGPT)
[![Hugging Face — Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/Ten1o/MouseGPT)
[![Hugging Face — Example data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Example%20data-FFD21E)](https://huggingface.co/datasets/Ten1o/MouseGPT-examples)

MouseGPT describes mouse posture and movement in natural language, then uses those descriptions to help researchers find behaviors, assign labels, and explore sessions. A vision-language model (VLM) reads images and text; an embedding model converts descriptions into numerical representations that can be compared during search.

This repository accompanies **[MouseGPT: A Large-scale Vision-Language Model for Mouse Behavior Analysis](https://arxiv.org/abs/2503.10212)** and contains the code for all four stages of the framework.

| Reader | Start here |
| --- | --- |
| **Researchers and neuroscientists** | Follow this README for a small example and a choice of workflows. No local GPU is needed for the hosted-model example. |
| **AI agents and automation developers** | Read [README_AGENTS.md](README_AGENTS.md) for execution order, data contracts, validation, and failure handling. |
| **Configuring a model provider** | Use [CONFIGURATION.md](CONFIGURATION.md) for model roles, endpoints, credentials, and compatibility. |

## Choose your starting point

| What you want to do | What you need | Guide |
| --- | --- | --- |
| **Try MouseGPT with a third-party VLM** | Python, Internet access, and access to a vision API and an embedding API | [Quick start below](#quick-start-use-a-hosted-vlm) |
| **Explore an existing example session** | Python and the prepared session; model APIs for conversation and model-assisted tools | [Natural-language interface](4_NLUI/readme.md) |
| **Use the MouseGPT-Large or MouseGPT-Lite weights locally** | A complete checkpoint and a compatible GPU environment | [Local model serving](2_VLM/readme.md#serve-a-mousegpt-model) |
| **Process your own recordings from the beginning** | Synchronized videos, camera calibration, detector and pose weights | [Pose and kinematics](1_keypoints/readme.md) |

**Hosted VLMs can be used now without downloading MouseGPT weights.** The local MouseGPT checkpoint uploads are pending; see the [model release checklist](https://huggingface.co/Ten1o/MouseGPT#todo-model-release). Model choice can affect description quality and downstream results. Inspect the descriptions against your behavioral definitions before using them in scientific comparisons.

## Quick start: use a hosted VLM

This example generates descriptions for three provided images, packages them as a small session, and searches the descriptions for a behavior. It uses a compatible external vision model and a text embedding model. They may come from different providers. The example demonstrates the software workflow; three frames are not an evaluation of behavioral accuracy.

Commands below run in a terminal from the repository root. Use **Python 3.12** for the documented environment. The hosted route runs on the CPU; it does not install the GPU training or pose-estimation stacks.

### 1. Download the code and install the environment

```bash
git clone https://github.com/Ten1o/MouseGPT.git
cd MouseGPT
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-api.txt
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and activate it with `.venv\Scripts\Activate.ps1`. Run subsequent `python` commands in that activated environment.

### 2. Download and inspect the small example

```bash
python scripts/download_assets.py vlm-demo
python scripts/download_assets.py vlm-demo --verify
python 2_VLM/prompt_test/client_test.py --validate-only
```

The download is approximately **0.29 MB** and contains **10 images and 10 prompt/response records**. The last command should report `Validated 10 prompts and 10 image references.` These checks run locally. Stored example responses are illustrative and are never sent as answers to the model.

### 3. Select your model services

Make a copy of `.env.example` named `.env.local` in the repository root. Keep an existing `.env.local` if you already have one. Open it in a text editor and fill in your settings. This file is excluded from Git.

For the default OpenAI configuration, fill in `MOUSEGPT_API_KEY`. To use another vision service while keeping the default embedding service, use the settings below. Edit any existing lines with these names and add only missing ones; each setting must appear once. Replace the example values with your own:

```dotenv
MOUSEGPT_API_KEY=YOUR_OPENAI_KEY
MOUSEGPT_VISION_PROVIDER=openai_compatible
MOUSEGPT_VISION_ENDPOINT=https://YOUR_PROVIDER_BASE_URL/v1
MOUSEGPT_VISION_API_KEY=YOUR_VISION_PROVIDER_KEY
MOUSEGPT_VISION_MODEL=YOUR_IMAGE_CAPABLE_MODEL_ID
```

The vision endpoint must support OpenAI-compatible chat completions with image input. Enter the provider's **base URL**, without `/chat/completions`. Built-in presets, including Azure OpenAI, Claude, Gemini, and other services, are explained in the [configuration guide](CONFIGURATION.md#endpoint-examples). A text-only language model cannot fill the vision role. You can also choose a separate embedding service with the `MOUSEGPT_EMBEDDING_*` settings.

Check the two roles needed for this example:

```bash
python -m mousegpt_config --role VISION
python -m mousegpt_config --role EMBEDDING
```

These commands check local settings without contacting a provider. The following generation and search steps send images, prompts, descriptions, or queries to the selected services and may incur API charges. Start with the three-frame example before processing a full session.

### 4. Generate new behavioral descriptions

```bash
python 2_VLM/prompt_test/client_test.py \
  --backend api --limit 3 --output-dir runs/api-demo/demo
```

The client uses the configured `VISION` model. It writes one validated description per frame under `runs/api-demo/demo/descriptions/` and matching JPEG previews under `runs/api-demo/demo/images/`. Each description contains `Overall`, `Head`, `Limb`, `Torso`, `Others`, and `Keywords`. Inspect these files and the images together before continuing. Use a new output directory when comparing models or runs.

### 5. Create a searchable session

An embedding represents the meaning of a description as numbers. The next commands generate these representations and store the descriptions, vectors, and a clustering hierarchy together in one HDF5 session file:

```bash
python 3_applications/embedding_request.py \
  --data_folder runs/api-demo/demo --max_workers 1
python 3_applications/create_h5.py \
  --workspace runs/api-demo --group_list demo --with-linkage
python scripts/query_session.py \
  --session runs/api-demo/demo.h5 --query "mouse walking" \
  --top-k 3 --output runs/api-demo/search.csv
```

The session file is `runs/api-demo/demo.h5`. Search prints frame IDs ranked by similarity to your query and writes `runs/api-demo/search.csv`, allowing you to open the corresponding previews. Choose a new CSV filename to retain results from another search. Similarity is a ranking, not a calibrated probability that a behavior occurred. The query must use the same embedding configuration that created the stored vectors; incompatible configurations are rejected.

To explore this session through chat, follow [Open your own session in the interface](4_NLUI/readme.md#use-your-own-session). To process all ten examples, use a fresh output directory and omit `--limit 3`. For your own paired camera views and measured kinematics, follow [Stage 2 input preparation](2_VLM/readme.md#prepare-images-and-kinematic-prompts).

## How the four stages connect

| Stage | Input → output | Guide |
| --- | --- | --- |
| **1. Keypoints and kinematics** | Synchronized videos and calibration → detected animals, body keypoints, and 3D motion measurements | [Pose pipeline](1_keypoints/readme.md) |
| **2. Vision-language descriptions** | Images and prompts, optionally with measured kinematics → structured behavioral descriptions | [VLM pipeline](2_VLM/readme.md) |
| **3. Behavior analysis** | Descriptions → embeddings, session files, categories, and cohort analyses | [Analysis workflows](3_applications/readme.md) |
| **4. Natural-language interface** | A prepared session → conversational search, classification, clustering, and frame exploration | [Interface guide](4_NLUI/readme.md) |

You can start at a later stage when you already have its inputs. The [examples guide](examples/README.md) explains the small VLM bundle and the prepared `240419_C11M4` session. The latter downloads approximately **985 MB**, expands to **1.39 GB**, and contains **5,999 stored frame records** with **6,000 JPEG previews**.

The [benchmark toolkit](benchmark/README.md) provides frame-description evaluation and [figure reproduction](benchmark/figure3/README.md). Training, pose reconstruction, and cohort analyses require their own data and dependencies; they are not prerequisites for the hosted-model quick start.

## Common setup questions

| What you see | What to do |
| --- | --- |
| `python3.12` is not found | Install Python 3.12, then create and activate the environment again. Confirm `python --version` before installing dependencies. |
| A configuration check asks for a key or model | Edit the repository-root `.env.local`; check the named role with `python -m mousegpt_config --role ROLE`. Each service needs its own credentials. |
| HTTP 401 or 403 | Check the selected service's key, account access, and endpoint. Repeated retries will not fix authentication. |
| The provider rejects images or a model name | Select a model with image-input support and use the provider's OpenAI-compatible endpoint. Basic chat support is insufficient. |
| A description fails validation | Inspect the response and input. The model must return all six required text fields; do not replace missing fields with invented values. |
| Search reports incompatible or missing embedding provenance | Recreate embeddings and the session using the intended embedding configuration. Matching vector lengths alone do not make models interchangeable. |
| A session has descriptions but cannot cluster | Package it with `--with-linkage`, or supply a hierarchy over the exact stored feature rows; see [Stage 3](3_applications/readme.md). |

For an unfamiliar input directory, `python scripts/check_workflow.py --input /path/to/session` suggests a starting stage. It does not replace the stage's data validation or a model test. For additional help, open a [GitHub issue](https://github.com/Ten1o/MouseGPT/issues) with the failing command, error, operating system, and Python version. Remove API keys and unpublished data before sharing logs.

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
