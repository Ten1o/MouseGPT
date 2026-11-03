# MouseGPT execution guide for AI agents

Use this guide when setting up, running, or modifying MouseGPT on behalf of a user. The [researcher README](README.md) explains the scientific workflow in plain language. [CONFIGURATION.md](CONFIGURATION.md) is the shared reference for provider fields and precedence; stage guides describe specialist workflows.

## Execution order

1. Identify the requested result and the available inputs. Select the relevant route below; do not run preceding stages unnecessarily.
2. Inspect the source revision, working tree, interpreter, and environment. Preserve user changes, existing configuration, input data, and previous results.
3. Validate local inputs and the required model roles before provider requests. Use the user's authorized destination and scope; start a newly configured workflow with a small representative input.
4. Execute with bounded attempts. Preserve frame IDs, model provenance, and outputs between stages.
5. Check the resulting files and requested downstream behavior. Report what was executed, the evidence of success, and anything untested. A passing configuration check is not proof of provider access or scientific accuracy.

## Repository and environments

Unless a command says otherwise, its working directory is the repository root, containing `mousegpt_config.py` and the four numbered stage directories.

```bash
git status --short
git rev-parse HEAD
python --version
```

Use **Python 3.12** for the documented CPU API environment. Do not assume the system interpreter is compatible. On a POSIX shell:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-api.txt
```

On Windows PowerShell use `py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1`. Check the active interpreter after activation. Use the relevant environment rather than installing every stage together:

| Route | Dependency contract |
| --- | --- |
| Hosted VLM → embeddings → session → query | Root `requirements-api.txt`; no GPU or local checkpoint |
| Natural-language UI | Add `4_NLUI/requirements.txt`; validate with `python 4_NLUI/agent.py --check` |
| Offline development tests | Root `requirements-dev.txt` |
| Local MouseGPT inference | Separate CUDA/PyTorch environment; bundled SWIFT instructions in the Stage 2 guide |
| Pose estimation and reconstruction | Stage 1 environment, detector/pose weights, calibration, and compiled dependencies |
| Full cohort analysis | Stage 3 analysis dependencies and cohort inputs; the API requirements cover basic embedding/packaging, not every specialist analysis |

The historical Conda exports in Stages 3 and 4 contain platform-specific builds. Do not treat them as universal installers or silently change the current environment to reproduce those exports.

## Model configuration

Create `.env.local` from `.env.example` only if it does not already exist. Edit existing keys in place; add missing fields once and reject duplicate entries rather than appending a second setting. Keep credentials in that file or in the process environment. Do not print secret values, place keys in command-line arguments, or commit them. Use an existing approved credential mechanism; never substitute a different account or destination just to make a check pass.

| Role | Used for | Required capability |
| --- | --- | --- |
| `VISION` | Image descriptions | OpenAI-compatible chat completions accepting image input and returning the six description fields |
| `EMBEDDING` | Stored description vectors and search/classification queries | Embeddings API returning finite, nonzero vectors of a consistent dimension |
| `ANALYSIS` | Model-assisted behavior categories and clustering | Chat completions with the task's parseable response schema |
| `AGENT` | Conversational interface | Chat completions with tool calling; plain chat success is insufficient |
| `PHENOTYPE` | Optional cohort prediction | Official OpenAI Responses API with Code Interpreter |

A role may use its own `MOUSEGPT_<ROLE>_PROVIDER`, `_ENDPOINT`, `_API_KEY`, `_MODEL`, and Azure `_API_VERSION`. A changed service requires its own key. See the [provider examples](CONFIGURATION.md#endpoint-examples), including the distinction between an Azure resource root and a compatible service base URL.

Resolution order is: supported CLI override → role environment variable → role `.env.local` value → shared environment variable → shared `.env.local` value → applicable legacy/default value. A shared shell value does not override a role-specific file value. Inspect the resolved non-secret summary for each role actually used:

```bash
python -m mousegpt_config --role VISION
python -m mousegpt_config --role EMBEDDING
```

Add `ANALYSIS`, `AGENT`, or `PHENOTYPE` checks when needed. Do not use `--allow-missing-key` as a production preflight. For a template-only inspection without credentials:

```bash
python -m mousegpt_config --config .env.example --allow-missing-key --ignore-environment
```

Configuration checks validate syntax and presence, not model access. Smoke checks are separate:

```bash
python scripts/download_assets.py vlm-demo
python scripts/download_assets.py vlm-demo --verify
python scripts/smoke_model_api.py --role VISION
python scripts/smoke_model_api.py --role EMBEDDING
```

The smoke commands above are dry runs. A configured role is still required; VISION needs the downloaded sample image or `--image PATH` for a JPEG/PNG of at most 1 MB. Add `--live` only within the authorized API run scope. Each live smoke command makes one request. It does not establish complete pipeline or agent-tool compatibility.

## Route A: hosted VLM example through a downstream task

Prerequisites: active API environment, valid `VISION` and `EMBEDDING` roles, and authorization for these requests. Use the repository root and a fresh output directory for a new configuration or input set.

```bash
python scripts/download_assets.py vlm-demo
python scripts/download_assets.py vlm-demo --verify
python 2_VLM/prompt_test/client_test.py --validate-only
python 2_VLM/prompt_test/client_test.py \
  --backend api --limit 3 --output-dir runs/api-demo/demo
python 3_applications/embedding_request.py \
  --data_folder runs/api-demo/demo --max_workers 1
python 3_applications/create_h5.py \
  --workspace runs/api-demo --group_list demo --with-linkage
python scripts/query_session.py \
  --session runs/api-demo/demo.h5 --validate-only
python scripts/query_session.py \
  --session runs/api-demo/demo.h5 --query "mouse walking" \
  --top-k 3 --output runs/api-demo/search.csv
```

Expected state:

- Input validation reports ten prompts and ten image references. `--limit 3` selects three records for inference; stored example `response` values are never model inputs.
- `runs/api-demo/demo/descriptions/` contains three valid JSON descriptions. `images/` contains three JPEG previews with matching frame IDs. Raw responses and `manifest.json` record the completed VLM run and its non-secret provenance.
- `feature/` contains one `<FRAME_ID>_des.csv` per description, with matching `.embedding-cache/<FRAME_ID>.json` metadata. No frame is omitted or duplicated.
- `runs/api-demo/demo.h5` contains aligned `descriptions`, `features_des`, and `linkage_matrix` tables. The vector table carries embedding identity; the hierarchy describes the exact stored row order.
- Local session validation succeeds. Query results contain only stored frame IDs, with finite similarity scores ordered from highest to lowest; `search.csv` contains the exported result.

This sequence checks software integration. A live run establishes only the exact configured provider/model combination. Compare generated descriptions with the images before treating them as scientific observations; similarity scores are not calibrated behavioral probabilities.

When repeating the sequence, choose a new query CSV path or omit `--output` to print results without writing another file. Existing CSV outputs are never overwritten.

The VLM API client resumes unchanged completed records. Changed provider/model settings, changed inputs, or modified outputs require a fresh output directory. Do not delete a provenance file or rename outputs to bypass this protection. Embedding generation can regenerate stale cached vectors; repackage the HDF5 after regeneration. Do not query an old session after changing embedding configuration.

## Route B: natural-language interface

Install the UI dependencies in the chosen compatible environment:

```bash
python -m pip install -r 4_NLUI/requirements.txt
```

For the published prepared session:

```bash
python scripts/download_assets.py nlui-demo
python scripts/download_assets.py nlui-demo --verify
```

The archive downloads about 985 MB and expands to 1.39 GB; ensure available disk space. It supplies `4_NLUI/data/240419_C11M4.h5` and `240419_C11M4_imgs/`. The legacy vectors are recognized only for the exact published HDF5 bytes and their original official OpenAI `text-embedding-3-large` configuration. A different embedding service requires regeneration and packaging; equal dimensions do not establish compatibility.

For a session created by Route A, install a copy for the UI only when that destination does not already exist:

```bash
mkdir -p 4_NLUI/data
cp runs/api-demo/demo.h5 4_NLUI/data/demo.h5
cp -R runs/api-demo/demo/images 4_NLUI/data/demo_imgs
```

Do not overwrite or merge an existing session directory. Select a new session name when needed. Configure `AGENT` with a tool-capable model, retain the session's `EMBEDDING` configuration, and configure `ANALYSIS` for clustering. After installing the session and configuring these roles, validate local startup and launch:

```bash
python 4_NLUI/agent.py --check
python 4_NLUI/agent.py
```

The UI is at `http://127.0.0.1:7860`. Select a session and exercise the requested tool, such as “Find frames showing grooming.” Inspect the returned frame IDs and exported files under `4_NLUI/workspace/`. Starting the server or receiving a chat greeting does not validate tool execution.

The interface includes a local Python execution tool. Keep it in a trusted local session. Tracing is off by default; enable it only when the user's workflow calls for it.

## Route C: browser workbench

The [UI guide](ui/README.md) covers `python ui/server.py --port 8878`. Run it from a checkout containing `ui/`; processing uses validated, explicitly selected environments on the backend machine. API settings are saved separately in the application's configuration directory. Preserve those settings and the user's source files. A stale environment validation must be refreshed before processing; do not manufacture a validation report or bypass the activation check.

The UI uses the same embedding generation, provenance-aware packaging and retrieval contracts as Route A. Packaging can read separate description/vector directories and writes a new HDF5 path. Enable **Clustering hierarchy** only when required, with the same quadratic-memory limitation as `--with-linkage`. Changing embedding models requires rebuilding vectors and the session.

Visual-review records belong to their canonical source directory and original frame ID, with a checksum of the source description. Load frames again after source changes. Save creates a new review snapshot; only that source session's reviewed frames are included, regardless of Passed/Needs correction status. Regenerate embeddings from the returned snapshot directory before packaging. Do not combine unscoped legacy drafts or reviews from another session.

For MMPose conversion, require consecutive integer frame IDs from zero and explicit `source_keypoints` in the mapping. Pose estimation needs an empty destination. Cancel terminates the worker and its children; inspect completed/partial outputs before retrying into a fresh destination. Inline Q&A uses retrieval; **Open tool-calling Agent** starts the separate repository Agent on port 7861.

## Input and output contracts

| Route | Required inputs | Acceptance criteria |
| --- | --- | --- |
| Hosted JSONL VLM | A `query` string and nonempty `images` array per record; paths relative to the JSONL; numeric first-image stem or explicit `frame_id` | API mode outputs matching eight-digit frame filenames; each JSON contains nonempty `Overall`, `Head`, `Limb`, `Torso`, `Others`, `Keywords` |
| Paired-camera VLM | `SESSION/pre_images/data3/`, `SESSION/pre_images/data5/`, `SESSION/velocity.json` | Nonempty matching camera-frame sets and matching measured velocities; every requested frame has a validated description |
| Embedding | Unique `descriptions/FRAME_ID.json` or `.txt` | One finite, nonzero vector per frame; consistent dimensions and verified per-vector metadata |
| Session packaging | Descriptions, vectors, and verified embedding sidecars | Exact frame correspondence; no missing/extra vectors; one identity for the session; atomic HDF5 output |
| Session search/classification | Aligned HDF5 descriptions/features plus matching configured embedding identity | Reject absent/mismatched identity before a query request; never compare incompatible vector spaces |
| Session clustering | HDF5 `linkage_matrix` over the exact feature rows | `N - 1` valid merges for `N` rows; singleton has an empty four-column table; unresolved labels remain `Unassigned` |
| Local SWIFT | SWIFT-compatible server, model checkpoint, JSONL prompts and images | Use `--backend swift`; its server key/protocol are separate from VISION. Output filenames need explicit frame mapping before session packaging |
| Raw-video processing | Synchronized recordings, calibration, detector and 24-keypoint pose weights | Preserve original camera frame indices; resolve units, nonfinite coordinates, and filtering before Stage 2 |

For paired-camera preprocessing, `tri_filter.npy` uses metres and one row per original video frame. Use the original frame rate. Interior velocities use the central difference divided by two; output velocities are cm/s. Never infer units or compute velocities from already-subsampled indices without accounting for elapsed time.

`create_h5.py --with-linkage` constructs average linkage using cosine distance. Its distance matrix has quadratic memory cost; use it for the small example or appropriately sized sessions. Cohort UMAP/clustering is a separate Stage 3 workflow with its own settings. A hierarchy generated under different row order or features must not be attached to the session.

## Provenance and failure handling

- Keep the VLM manifest, raw responses, validated descriptions, embedding sidecars, and HDF5 metadata together. Do not manufacture missing provenance for old files.
- Embedding identity includes the service, endpoint, model/deployment, API version, settings, and vector dimension. Regenerate embeddings and rebuild the session when any identity component changes. Arbitrary legacy HDF5/CSV data without verified provenance must be regenerated.
- Clustering caches depend on the selected analysis provider/model and the session inputs. Preserve cache metadata; do not present an earlier model's cached result as a new run.
- A nonzero exit, missing frame, malformed output, authentication failure, incompatible vector space, or unresolved prerequisite blocks the dependent step. Report the actual error and fix its cause; do not suppress exceptions to continue.
- Retry only within the script's finite limits. Do not retry authentication errors indefinitely or automatically expand a paid run after a failed smoke check.
- `scripts/check_workflow.py` suggests an entry point from filenames. Its output is advisory; it does not inspect every file's contents or replace the validation commands above.

## Development checks and completion report

In the development environment, run:

```bash
python -m pip install -r requirements-dev.txt -r 4_NLUI/requirements.txt
python -m pytest tests 4_NLUI/tests -q
python -m mousegpt_config --config .env.example --allow-missing-key --ignore-environment
git diff --check
```

For workbench changes, install `ui/requirements-test.txt` and also run `python -m pytest ui 1_keypoints/tests/test_batch_vis_ui.py 1_keypoints/tests/test_convert_sleap.py 1_keypoints/tests/test_reprojection_ui.py -q` and `node --test ui/tests/review_state.test.cjs` (Node.js 22). These checks use temporary data and simulated provider responses. See the CI workflow for the complete platform matrix.

The test suite uses controlled provider responses and does not require production keys or downloaded examples. Report skipped tests and their reasons. Use representative regression tests for changes to provider routing, credential isolation, vector provenance, caching, frame alignment, and pipeline outputs. Do not claim live provider coverage based on mocked tests.

A completion report should state the source revision, environment, route, input/output locations and counts, selected non-secret model identities, commands/checks that passed, and any failures or untested components. Distinguish local validation, mocked integration, live model requests, UI startup, and actual UI tool execution. State scientific limitations separately from software execution results.
