# MouseGPT UI User Guide

MouseGPT UI provides keypoint processing, behavior descriptions, behavior analysis and Agent interaction in a desktop browser. Tasks run on the machine hosting the UI service and save results to the paths you specify.

## 1. Open the UI

Install **Git** and **Python 3.10 or later**. On Windows, Python must be available to the launcher. Opening the UI does not require Node.js, npm, Conda or the project's training dependencies.

### Windows

Run in PowerShell:

~~~powershell
git clone https://github.com/Ten1o/MouseGPT.git
cd MouseGPT
.\ui\start-ui.cmd
~~~

Keep the terminal open and visit **http://127.0.0.1:8878/ui/**. You can also double-click ui/start-ui.cmd. The launcher prefers Python 3.13 at its default installation location, then falls back to python in the terminal.

### macOS / Linux

~~~bash
git clone https://github.com/Ten1o/MouseGPT.git
cd MouseGPT
python3 ui/server.py --port 8878
~~~

Open the same URL. The UI supports desktop browsers only; mobile devices are not supported.

To test an unmerged UI pull request, check out its branch first. The commands above require a branch containing ui/.

**Opening the UI does not configure its processing environments.** Complete the environment and API setup below before running tasks. Functions that do not call an API do not need an API key.

## 2. Initial setup

### Environments

Open **Environments** from Overview or the sidebar.

| Environment | Functions | Setup |
| --- | --- | --- |
| Core workflow · CPU & API | Image/velocity preprocessing, format conversion, filtering/visualization, API descriptions, embeddings, HDF5, behavior analysis and inline retrieval Q&A | Install & validate, or validate an existing environment |
| Tool-calling Agent | Stage 4 Open tool-calling Agent, including the original Gradio/LangChain application | Install & validate, or validate an existing environment |
| Pose estimation & reconstruction | Stage 1 pose estimation and triangulation | Install dependencies and models manually, then validate; automatic installation is deferred |

1. Install a new environment, or enter the full Python executable path under **Use an existing environment** and validate it.
2. A successful check produces a **Validated** card.
3. Click **Use**. The green **Active** status indicates that the environment is selected for its tasks. After an application update, a **Revalidate** status means the environment must pass the current checks again. Agent validation builds the actual application with a temporary session and no provider requests.

Validated means checks passed; Active means selected. Each of the three environment profiles can be active independently. You do not need to repeatedly activate Conda in a terminal. Advanced interpreter settings may select another validated environment; task access is checked against the interpreter actually used.

Automatic installation targets Windows x64, Linux x64/ARM64 with glibc ≥ 2.28, and macOS 14+ on Intel/Apple Silicon. The current Apple Silicon Agent recipe has passed a fresh managed installation and application startup check. The other updated Agent recipes are resolved but await runtime acceptance; earlier Windows/Ubuntu checks apply to their recorded releases. See [environment installation](environments/README.md) for the validation scope. Installation requires internet access and at least 3 GB of free disk space, plus space for your data.

For manual Pose dependencies and models, see the [Stage 1 guide](../1_keypoints/readme.md). Supply matching detection/pose .py configurations and .pth weights. Cloning the repository does not download weights, videos, camera calibrations or a local EasyMocap installation.

### API settings

Open **API settings** from Overview or the sidebar.

1. Configure the provider, API key, model and service URL under **Default API**. Azure also requires its resource endpoint, deployment name and API version.
2. Enable **Custom API** only for roles needing a different configuration.
3. Click **Save configuration**. Use **Save & test** when you want to send a real connection-test request.

| API role | Functions |
| --- | --- |
| VISION | Stage 2 behavior descriptions from images and motion information |
| EMBEDDING | Description embeddings for analysis and retrieval |
| ANALYSIS | Cluster naming, fine-grained labels and related analysis |
| AGENT | Stage 4 Q&A and the full Agent |
| PHENOTYPE | Stage 3 behavioral phenotype prediction |

Embedding requires an embedding model rather than a chat model. The default is **text-embedding-3-large**. Azure users must select their embedding deployment.

PHENOTYPE currently requires the official OpenAI Responses API with Code Interpreter. Other roles can continue using Azure or compatible services. You do not need a separate PHENOTYPE configuration if you do not use phenotype prediction.

**Import env / JSON** accepts project-format .env.local, .env.example and UI-exported JSON. Import fills the form; click Save to apply it. The .env.example file is a template and needs your own keys and models. Exports omit keys unless **Include API keys (plaintext)** is selected. Leaving a key field blank preserves its saved key only when its provider and endpoint are unchanged. When switching services, enter that service's key; the previous credential is never reused automatically.

API tasks send the relevant images, descriptions or statistics to the selected service and may incur charges. Keys are stored in the backend machine's application configuration; do not commit them to GitHub.

## 3. Available workflows

Start at the Stage matching your existing data. Each Stage contains independently runnable steps.

| Page | Main inputs | Functions and outputs |
| --- | --- | --- |
| Stage 1 · Keypoints & 3D reconstruction | Eight synchronized data1.mp4 … data8.mp4 videos and matching models/configurations, or existing 2D keypoints; eight camera MAT files | Pose estimation/conversion → data1.json … data8.json; triangulation → tri_new.npy; filtering/unit conversion → tri_filter.npy and a full-length 3D MP4; reprojection → eight-view JSON and overlay MP4 files |
| Stage 2 · Images & behavior descriptions | Camera 3/5 videos, keypoints and FPS; or camera 3/5 crop JPGs and velocity.json | Frame extraction, cropping, velocity calculation and paired previews; per-frame description JSON with an editable prompt; visual review and revision records |
| Stage 3 · Behavior analysis & session packaging | Description JSON and corresponding embeddings, or a session .h5 | Embedding CSV and session HDF5; keyword analysis, major behavior clustering, fine-grained classification, novel behavior candidates and temporal statistics; phenotype prediction additionally requires cohort/feature files |
| Stage 4 · Session Q&A | Session .h5 and a directory of frame-matched JPGs | Session checks, automatic slider previews, semantic retrieval Q&A and the separate full tool-calling Agent |

### Files and paths

- Inputs and outputs may be outside the project directory; you do not need to move them into workspace.
- Output-directory fields take directories. **New session HDF5 path** takes a complete new filename, such as /data/results/240419_C11M4.h5.
- Most outputs cannot overwrite existing files. Choose a new output path when rerunning; do not delete input data.
- Task configurations can be exported and imported.

### Stage 1: data requirements

The triangulation workflow expects eight synchronized views, 24 keypoints and one selected animal per frame. Unified JSON and camera calibration ordering must match. Conversion supports single-animal DLC CSV/HDF5 and MMPose JSON with camera/keypoint-name mapping. SLEAP Analysis HDF5 exports are also supported (see below).

For MMPose, the Mapping JSON must include `source_keypoints`: 24 names in the exported node order, either one list shared by all cameras or an object mapping each camera to its list. The existing `keypoints` list specifies the target order. Every camera must contain exactly one animal per frame and consecutive integer `frame_id` values starting at zero. Sparse, offset, duplicated or reordered frame sequences are rejected. Pose estimation requires an empty output directory; use a new directory for another run.

For SLEAP, export one Analysis HDF5 file per camera using **File > Export Analysis HDF5** in SLEAP. Native .slp projects are not read directly. Both the original MATLAB layout and the sleap-io standard layout are accepted without installing SLEAP in the Core environment. Map each camera to its .analysis.h5 filename and list 24 node names in the required target order.

Single-track files are selected automatically. For multiple tracks, add a tracks object to Mapping JSON with a track name or zero-based index for each camera, for example: "tracks": {"data1": "mouse0", "data2": 0}. Select the same animal across all views. Coordinates must use calibrated original-image pixels; exports must cover matching frames from frame 0. Missing points retain their frames and receive confidence 0. Files without point scores use confidence 1 for finite points.

For filtering, select tri_new.npy and confirm its original unit. The output tri_filter.npy uses metres and preserves the original frame correspondence. The 3D animation includes all frames by default.

Reprojection can reuse earlier paths or accept separate NPY, video and calibration paths. Select **Metres** for tri_filter.npy and the actual translation unit of your calibration. The NPY and each video must correspond from original frame 0 and have matching frame counts. Select a camera to view its completed overlay video.

### Stage 2: descriptions and visual review

Enter the actual video FPS. Preprocessing expects 3D coordinates in metres; velocity JSON uses cm/s. Description generation defaults to a small frame sample. **Frame limit = 0** processes all frames.

**Visual review** displays images, velocities and description items for assessment. Drafts are separated by the canonical source directory and original frame ID. Switching source directories does not transfer corrections. If a source file changes, load it again; old drafts remain archived and are not applied to the new content. Older drafts without a source identity are retained for export in the task configuration, but must be reviewed again before saving.

**Save review** creates a new snapshot under the output directory's `reviews/` folder, with review records and a `descriptions/` copy. It preserves source files and previous snapshots. Only the current session's reviewed frames are included. Check that frame set before embedding the **Reviewed copy**: it includes all saved review statuses, not only frames marked Passed. Generate embeddings for the saved copy before packaging it.

### Stage 3: sessions and analysis results

Packaging requires matching original frame IDs and verified embedding sidecars for every description. It accepts a consistent vector dimension from the selected embedding model, validates the source and vector hashes, and stores the model identity in the session. Stale, mixed or undocumented vectors must be regenerated. Search requires the same embedding configuration; matching vector lengths alone is insufficient.

Choose **Clustering hierarchy → Yes** when the session should support the full Agent's clustering tool. This builds average linkage with cosine distance over the stored feature rows and requires memory proportional to the square of the number of frames. Leave it off for search-only sessions or large datasets requiring a separate clustering workflow. UMAP needs at least three valid records; meaningful clustering requires a sufficient sample.

Clustering and fine-grained analysis write a **new HDF5 copy** in the results directory. They do not add labels to the input session. Use the output path reported by the task for subsequent analysis.

The Temporal analysis **HDF5 label table** is an internal table name, not a filesystem path:

| Classification result | Table name |
| --- | --- |
| Major behavior profiling output HDF5 | label_All_cluster_ui |
| Fine-grained Head output HDF5 | label_Head_keyword_ui |
| Fine-grained Limb / Torso output HDF5 | label_Limb_keyword_ui / label_Torso_keyword_ui |
| Fine-grained Overall / Others output HDF5 | label_Overall_keyword_ui / label_Others_keyword_ui |

Also enter the original FPS and frame sampling interval. Temporal results use sampled labels and do not identify exact continuous-video behavior boundaries.

### Stage 4: previews and Agent

The image directory contains files such as 00000000.jpg and 00000012.jpg matching original session frame IDs. The current example pairs camera 3 and camera 5 crops. The slider selects existing images and loads them automatically; you can also enter a frame number.

Inline **Session Q&A** retrieves behavior descriptions. **Open tool-calling Agent** launches the original Agent and requires the separate Agent environment. Open it at **http://127.0.0.1:7861/**.

## 4. Progress and troubleshooting

Task status beside each processing module shows progress, logs, errors and result files. Tasks at the bottom lists tasks from the current server session. One processing task runs at a time. Cancel stops the task and its child processing programs before another task can start. Previously completed files remain; inspect partial output and choose a fresh destination when rerunning. Restarting clears task history but preserves files already written.

- **Page unavailable:** Keep the service terminal open and check port 8878. Ctrl+C stops the service; KeyboardInterrupt alone does not indicate a processing failure.
- **Address already in use:** Identify and stop the old UI service before restarting.
- **Environment required:** Validate the appropriate environment and click Use.
- **No such file or directory:** Check paths on the backend machine and retain the complete Stage directories. Copying only ui/ is insufficient.
- **Old UI after an update:** Restart the service and refresh with Ctrl+F5.

The UI listens on localhost by default. For remote use, forward ports through SSH from your computer. Code, environments and data must be accessible on the server. Replace the user, address and SSH port below:

~~~bash
ssh -p 22 -N -L 8878:127.0.0.1:8878 -L 7861:127.0.0.1:7861 user@server
~~~

Keep the tunnel open and visit the same localhost URLs on your computer. API requests originate on the backend, which must be able to reach the selected service.

## 5. Validation and supported scope

Automated checks cover session-scoped reviews, credential changes, process cancellation, embedding provenance, packaging and retrieval, synthetic keypoint conversion and video processing, and the actual Agent's local startup. Provider responses in workflow tests are simulated. See [environment installation](environments/README.md) for recipe-specific runtime verification. Successful local validation does not establish scientific accuracy or compatibility with every OS, provider, model or input format.

**Incomplete or deferred:** Automatic Pose/GPU environment installation, local VLM installation/training/serving through the UI, and standalone desktop packaging. Review items currently use punctuation-based draft splitting rather than semantic fact extraction. Mobile devices are not supported.

**Additional validation needed:** Representative real DLC/MMPose/SLEAP exports and mappings; complete workflows on each target platform; additional provider/model combinations; and prolonged low-memory operation. Check generated descriptions and measurements against the underlying recordings before scientific comparisons.

For development checks, install `ui/requirements-test.txt`, then run the repository tests and UI tests with `python -m pytest tests 4_NLUI/tests ui -q`. Node.js 22 is used only for browser-logic tests (`node --test ui/tests/review_state.test.cjs`); running the workbench does not require Node.js.
