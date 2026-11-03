# Stage 2 · Vision-language behavior description

MouseGPT combines mouse images with kinematic information to generate structured behavior descriptions. This directory provides the inference client, model-serving integration, image preprocessing, annotation utilities, and training dataset registration.

**Inputs:** mouse images and natural-language prompts, with optional measured kinematic context.

**Outputs:** model-generated descriptions for downstream behavior analysis.

[Model repository](https://huggingface.co/Ten1o/MouseGPT) · [Example dataset](https://huggingface.co/datasets/Ten1o/MouseGPT-examples) · [Stage 3: behavior analysis](../3_applications/readme.md)

> **Choose a model route:** use a hosted, image-capable API with the `VISION` configuration, or serve MouseGPT weights locally with SWIFT. The hosted route needs no local GPU or MouseGPT checkpoint. Local checkpoint uploads are forthcoming; only the local-serving route requires those weights.

## Inspect the example inputs

Run from the repository root:

```bash
python scripts/download_assets.py vlm-demo
python 2_VLM/prompt_test/client_test.py --validate-only
```

This downloads `test_data.jsonl` and ten images into `2_VLM/prompt_test/data/`. Each example contains a `query`, an `images` list, and an illustrative `response`. Image paths are relative to the JSONL file. The inference client sends only the query and images; the stored response is not included in the model input.

Input validation uses the Python standard library and does not require a GPU or model installation.

## Use a hosted VLM

For a complete image-to-search example, follow the [researcher quick start](../README.md#quick-start-use-a-hosted-vlm). The hosted client reads `VISION` from the repository-root `.env.local`; it supports services exposing OpenAI-compatible image chat completions, including Azure deployments and the provider presets in [CONFIGURATION.md](../CONFIGURATION.md).

From the repository root, with the API environment active:

```bash
python -m pip install -r requirements-api.txt
python scripts/download_assets.py vlm-demo
python -m mousegpt_config --role VISION
python 2_VLM/prompt_test/client_test.py \
  --backend api --limit 3 --output-dir runs/api-demo/demo
```

This sends the selected prompts and images to the configured provider and may incur API charges. The `response` field in the downloaded JSONL is never included in the request. The model must return nonempty strings for `Overall`, `Head`, `Limb`, `Torso`, `Others`, and `Keywords`.

The output is ready for [Stage 3 session packaging](../3_applications/readme.md#build-a-session-from-your-descriptions):

```text
runs/api-demo/demo/
├── descriptions/       # Validated FRAME_ID.json files
├── images/             # Matching FRAME_ID.jpg previews
├── raw_descriptions/   # Original provider response per frame
└── manifest.json       # Model identity and input/output hashes
```

The first image's numeric filename stem becomes the eight-digit frame ID. For another JSONL, provide an explicit numeric `frame_id` when image names are not numeric. IDs must be unique. A record with several images keeps all of them in the model request; the first image supplies its preview. Do not substitute row numbers for original camera frame IDs when processing experimental recordings.

| API-client option | Meaning |
| --- | --- |
| `--backend api` | Use the configured VISION service. The default backend remains `swift` for local deployment compatibility. |
| `--input PATH.jsonl` | Read prompts and image paths from another JSONL. Relative image paths resolve from this file. |
| `--limit N` | Run the first N records; start with a small sample. |
| `--output-dir PATH` | Session output directory. Use a new location for a new model or input set. |
| `--model MODEL_ID` | Explicitly override the VISION model/deployment. |
| `--max-attempts N` | Finite attempts per record; default 3. Authentication and other non-transient rejections stop without repeated requests. |
| `--timeout SECONDS` | Request timeout. |
| `--max-tokens N` | Completion budget; default 2,000. Some reasoning models spend part of this budget on reasoning. |
| `--validate-only` | Validate local prompt records and image paths without inference. |

Unchanged completed records can resume. The client rejects reuse of an output directory when model configuration, inputs, or completed outputs differ from its manifest. It also rejects incomplete or malformed descriptions; failed model responses are saved separately under `failed_responses/`, with the diagnostic path reported by the command. They are never treated as completed frames. Inspect a failure before retrying; if a reasoning model exhausts its completion budget, choose a larger budget in a new run. `--base-url` and `--api-key-env` belong to the local SWIFT route; choose hosted endpoints and credentials through the VISION configuration.

## Serve a MouseGPT model

### Install the model environment

Use a dedicated Linux GPU environment with CUDA-compatible PyTorch. From the repository root, install the bundled SWIFT source, which includes the MouseGPT dataset integration:

```bash
python -m pip install -e './2_VLM/swift[llm]'
```

The bundled version is `2.4.0.dev0`. Its [dependency requirements](swift/requirements/) include `numpy<2.0` and `transformers>=4.33,<4.45`. Install an LMDeploy backend for MouseGPT-Large or a vLLM backend for MouseGPT-Lite compatible with this SWIFT version, the model architecture, and your PyTorch/CUDA environment. Backend and CUDA versions are not pinned by this repository.

### Start the server

Once checkpoints are available from the [model repository](https://huggingface.co/Ten1o/MouseGPT), replace the paths below with the corresponding local checkpoint directories, including their configuration files. Start either server:

```bash
# MouseGPT-Large
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift deploy \
  --ckpt_dir /path/to/MouseGPT-Large \
  --infer_backend lmdeploy --tp 8
```

```bash
# MouseGPT-Lite
CUDA_VISIBLE_DEVICES=0 swift deploy \
  --model_id_or_path /path/to/MouseGPT-Lite \
  --model_type minicpm-v-v2_6-chat \
  --infer_backend vllm
```

These examples allocate eight GPUs to Large and one GPU to Lite. Required GPU memory depends on the checkpoint, precision, context length, and inference backend. Both examples use the default server port, `8000`.

## Run local SWIFT inference

In a second terminal, run from the repository root:

```bash
python -m pip install -r 2_VLM/prompt_test/requirements-client.txt
python 2_VLM/prompt_test/client_test.py \
  --backend swift --base-url http://localhost:8000/v1 --limit 1
```

Remove `--limit 1` to process all examples. The client selects the model automatically if the endpoint advertises exactly one model; otherwise specify `--model <served-model-id>`. Set `MOUSEGPT_SERVING_API_KEY` if your SWIFT server requires authentication; this is separate from the repository-wide `MOUSEGPT_API_KEY` used for external model APIs.

Images are transmitted as base64 using the bundled SWIFT protocol. The server does not need access to the client's filesystem. The endpoint must support SWIFT's `images` request field.

| Option | Purpose |
| --- | --- |
| `--input /path/to/data.jsonl` | Read another JSONL file with the same input format. |
| `--output-dir /path/to/results` | Choose where to write results. |
| `--model <served-model-id>` | Select a specific model advertised by the server. |
| `--limit N` | Process the first `N` records. |
| `--save-previews` | Save an overlay on the first image of each record; requires `opencv-python`. |
| `--validate-only` | Validate records and local image paths without contacting the server. |

By default, descriptions are written to `2_VLM/prompt_test/output/output_des/`; optional previews go to `output/output_img/`. Filenames combine the record number and image stem. Reusing the same output directory replaces files with matching names, so choose a separate directory to retain another run.

## Prepare images and kinematic prompts

The utilities in [data_prepareation/](data_prepareation/) prepare synchronized camera views and kinematics for annotation. Run the commands below from the repository root. Image preprocessing requires NumPy, SciPy, Pillow, and tqdm; video extraction requires FFmpeg 6 or newer.

### 1. Extract synchronized frames

Place camera recordings `data3.mp4` and `data5.mp4` in the input directory. For example, to extract from minute 0 through the end of a 59.94 fps recording:

```bash
python 2_VLM/data_prepareation/vids_to_imgs.py \
  --input_folder /path/to/videos \
  --output_folder /path/to/session \
  --start_time 0 --end_time 0 --fps 59.94
```

Replace `--fps` with the original constant frame rate of both recordings. Times are specified in whole minutes. The script samples every 12 frames, starting at `int(fps * 60 * start_time)`, and writes images under `images/data3/` and `images/data5/`. Filenames retain the exact zero-based source frame index. A nonzero `end_time` is exclusive after conversion to a frame index; `0` means the end of the recording.

Use a fresh output location. Existing camera image directories are not overwritten, and both videos must finish extraction before the images are made available in the destination directories.

### 2. Crop images and compute velocities

The keypoint directory must contain per-frame predictions `data3.json` and `data5.json`, including the first animal instance's bounding box, plus `tri_filter.npy`: a filtered 3D keypoint array in **metres**, with one row per original video frame. Apply filtering and any unit conversion to the [Stage 1 reconstruction](../1_keypoints/readme.md) before using this step.

```bash
python 2_VLM/data_prepareation/preprocess_imgs.py \
  --data_folder /path/to/session \
  --json_folder /path/to/keypoints \
  --fps 59.94
```

The script creates 512 × 512 image crops under `pre_images/data3/` and `pre_images/data5/`, and writes `velocity.json` in the session directory. Keypoint coordinates must be finite, and at least two frames are required. Keep image filenames aligned with the original frame indices even when images are subsampled.

**Velocity convention:** interior frames use `(position[t+1] - position[t-1]) * fps / 2`, converted from m/s to **cm/s**. The first and last frames use forward and backward one-sided differences. Body speed is the magnitude of keypoint 4's velocity. Head and limb speeds are relative to keypoint 4, using keypoints 2, 10, 14, 18, and 21 for the head, left/right forepaws, and left/right hindpaws, respectively.

### 3. Generate structured annotations

`prompt_request.py` sends synchronized image pairs and measured velocity context to the configured VISION API. Use the root `requirements-api.txt` environment and the same `.env.local` configuration as the hosted client. Select an image-capable model using `MOUSEGPT_VISION_MODEL` and the relevant provider settings:

```bash
python 2_VLM/data_prepareation/prompt_request.py \
  --data_folder /path/to/session --workers 1
```

Camera frame IDs and velocity records must match. Each validated description contains `Overall`, `Head`, `Limb`, `Torso`, `Others`, and `Keywords`. Raw API responses are saved in `raw_descriptions/`, and validated JSON objects in `descriptions/`.

The default limit is three attempts per frame for transient errors; configure it with `--max-attempts` and set concurrency with `--workers`. Invalid descriptions and non-transient HTTP errors are reported as failures. Cached responses are reused only when the model, prompt, images, velocity inputs, and saved response match their recorded hashes. Changed inputs or missing provenance trigger a new request. API keys are excluded from request logs.

## Train with your own data

The dataset registration hook, [custom_dataset.py](data_prepareation/custom_dataset.py), reads `MOUSEGPT_TRAIN_DATASET_ID`. Set it to your training dataset identifier before importing the hook. Records must contain `query`, `response`, and `images` fields.

Training additionally requires a base checkpoint, dataset splits, and a task-specific SWIFT configuration. The example dataset is for demonstrating inference and does not provide a full training set. See the bundled [SWIFT documentation](swift/docs/) for training APIs and configuration options. Dataset registration is not required for inference.

## Additional utilities

For inference using previously saved annotation prompts, supply an image list and the matching annotation-log directory:

```bash
python 2_VLM/prompt_test/prompt_test.py \
  --image-list data/images.txt \
  --log-folder data/annotation_logs \
  --output-folder output/log_inference \
  --base-url http://localhost:8000/v1
```

Image-list entries are relative to the list file, and logs use the matching image stem, such as `00000012.json`. This utility requires a SWIFT-compatible endpoint and OpenCV for its image previews. Use `--model` to select a served model explicitly. The separate `all_data_test.py` utility targets an external Azure deployment; use `client_test.py --backend api` for hosted VLMs or `--backend swift` for local MouseGPT serving.

To select cohort records from one or more JSONL files:

```bash
python 2_VLM/prompt_test/extract_jsonl.py data/train.jsonl data/test.jsonl \
  --groups C3 C4 C5 --output data/selected/records.jsonl
```

Omit `--groups` to include all records. Cohort matching uses complete IDs, so `C1` does not match `C10`. Relative image paths are adjusted to the destination JSONL location.
