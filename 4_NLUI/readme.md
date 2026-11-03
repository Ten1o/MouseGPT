# Stage 4 · Natural-language interface

Explore a MouseGPT session by asking questions about behavior. The interface combines stored descriptions, text embeddings, and frame previews to support search, label assignment, clustering, and unusual-frame discovery.

The prepared example runs without repeating pose estimation or VLM inference. Interface code lives in this GitHub repository; session data are distributed through [Hugging Face Datasets](https://huggingface.co/datasets/Ten1o/MouseGPT-examples).

## Installation

Use the Python 3.12 environment from the [researcher quick start](../README.md#quick-start-use-a-hosted-vlm). From the repository root, install the interface dependencies:

```bash
python -m pip install -r 4_NLUI/requirements.txt
```

Next, install a session and configure its model roles as described below. The historical `environment.yml` is retained as a reference for the earlier Conda environment; its platform-specific builds are not required by this setup.

## Download the example session

Install `nlui-demo` using the [example-data instructions](../examples/README.md). The downloader creates this layout inside `4_NLUI/`:

```text
data/
├── 240419_C11M4.h5
└── 240419_C11M4_imgs/
    └── *.jpg
```

The HDF5 file provides precomputed descriptions, embeddings, and a clustering hierarchy. The paired image folder supplies previews. Both are required to open a session. The exact published HDF5 is recognized by its file hash as using official OpenAI `text-embedding-3-large` vectors. Keep that EMBEDDING configuration for search/classification, or regenerate a new session from the descriptions using your chosen model.

## Launch the interface

Create the repository-root `.env.local` from `.env.example` if it does not already exist, and fill `MOUSEGPT_API_KEY` for the default official OpenAI API. The same file also configures Stage 2 and 3 requests; see [configuration](../CONFIGURATION.md). For another provider, configure the `AGENT` role with a tool-capable chat model. Configure `EMBEDDING` for search/classification and `ANALYSIS` for clustering. Then run from the repository root:

```bash
python 4_NLUI/agent.py --check
python 4_NLUI/agent.py
```

The check validates local configuration, session files, and interface construction without starting a web server or contacting a model. A successful check does not verify provider access.

Open **<http://127.0.0.1:7860>**, choose a session, and use the frame slider or chat to explore its behavior. By default, conversation uses `gpt-4o`; semantic search and label assignment use `text-embedding-3-large`. Queries and relevant descriptions are sent to the configured provider and may incur usage charges.

| Explore | Example prompt | Method |
| --- | --- | --- |
| Inspect a frame | “Show the description of this frame.” | Retrieve the stored description |
| Find a behavior | “Find frames showing grooming.” | Rank descriptions by embedding similarity |
| Assign labels | “Classify the frames as grooming, walking, or resting.” | Match embeddings to user-defined label descriptions |
| Discover groups | “Cluster the behavior in this session.” | Traverse the stored hierarchy with language-model merge decisions |
| Inspect unusual frames | “Which frames are most unusual?” | Rank stored embeddings with IsolationForest |

Search, classification, clustering, and outlier tools export their results under `4_NLUI/workspace/`. CSV results retain the HDF5 frame IDs so they can be linked back to session images. Conversation logs are stored under `4_NLUI/workspace/user_logs/`.

## Use your own session

Add `SESSION_ID.h5` and a matching `SESSION_ID_imgs/` folder under `4_NLUI/data/`. Preview filenames must use the stored frame ID, for example `00000012.jpg`.

For the session created by the [hosted-VLM quick start](../README.md#quick-start-use-a-hosted-vlm), run from the repository root. Check that `demo.h5` and `demo_imgs/` do not already exist; use another session name if they do:

```bash
mkdir -p 4_NLUI/data
cp runs/api-demo/demo.h5 4_NLUI/data/demo.h5
cp -R runs/api-demo/demo/images 4_NLUI/data/demo_imgs
python 4_NLUI/agent.py
```

Select `demo` in the interface. The `AGENT` model can differ from the VLM that wrote the descriptions. Keep the `EMBEDDING` configuration that created the session vectors. To use clustering, package with `create_h5.py --with-linkage` and configure an `ANALYSIS` model.

| HDF5 table | Required contents |
| --- | --- |
| `descriptions` | Unique `id` values and their `description` records |
| `features_des` | Finite, nonzero description embeddings, indexed by the same IDs in the same row order; verified embedding identity in table metadata |
| `linkage_matrix` | A hierarchy over those feature rows, required by the clustering tool |

Search and classification require the same service, model/deployment, settings, and dimensions used to create the stored vectors. They reject incompatible or unknown identity before sending a query request. The app validates description/feature alignment before loading a session and displays previews with matching stored records. See [Stage 3](../3_applications/readme.md) for description embedding, provenance, and session packaging.

The distributed example contains 5,999 description/feature rows and 6,000 JPEGs. One additional image, `00037368.jpg`, has no stored feature row and is excluded from the interface. Frame IDs define correspondence; image-list positions do not.

## Interpret the outputs

Similarity search returns the closest descriptions to a query. Label assignment selects among the supplied label descriptions; its categories therefore depend on the labels you choose. IsolationForest identifies unusual embeddings within the loaded session, which need not represent a distinct or rare behavior in a broader population.

Clustering labels only groups with a confirmed merge. Frames that remain unresolved at a leaf or traversal depth limit receive `Unassigned` (`-1`), and remain included in exported counts and proportions. Clustering caches include the selected analysis service and model; changing that configuration recomputes the relevant decisions. Language-model categories can vary between runs, so check them against the intended behavioral definitions before drawing quantitative conclusions.

## Runtime configuration

The interface listens on localhost and processes requests sequentially. Its Python REPL tool can execute code in the local process; use it in a trusted local session. Direct clustering and embedding requests use finite connection/read timeouts and at most three attempts.

LangSmith/LangChain tracing is disabled by default. To enable it, set `LANGCHAIN_TRACING_V2=true` and configure your tracing credentials before launch.

For new video and pose processing, start with [Stage 1](../1_keypoints/readme.md) and [Stage 2](../2_VLM/readme.md). Cohort-level phenotype analysis is available separately in [Stage 3](../3_applications/readme.md#phenotype-analysis).
