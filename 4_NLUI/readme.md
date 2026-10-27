# Stage 4 · Natural-language interface

Explore a MouseGPT session by asking questions about behavior. The interface combines stored descriptions, text embeddings, and frame previews to support search, label assignment, clustering, and unusual-frame discovery.

The prepared example runs without repeating pose estimation or VLM inference. Interface code lives in this GitHub repository; session data are distributed through [Hugging Face Datasets](https://huggingface.co/datasets/Ten1o/MouseGPT-examples).

## Installation

Create the stage-specific environment from the repository root:

```bash
cd 4_NLUI
conda env create -f environment.yml
conda activate swift-langchain
```

The supplied environment targets Intel macOS with Python 3.10.14 and includes platform-specific package builds. For Linux, Apple silicon, or another platform, adapt those builds while retaining compatible Gradio, LangChain, and scientific Python dependencies.

## Download the example session

Install `nlui-demo` using the [example-data instructions](../examples/README.md). The downloader creates this layout inside `4_NLUI/`:

```text
data/
├── 240419_C11M4.h5
└── 240419_C11M4_imgs/
    └── *.jpg
```

The HDF5 file provides precomputed descriptions, embeddings, and a clustering hierarchy. The paired image folder supplies previews. Both are required to open a session.

## Launch the interface

Copy the repository-root `.env.example` to `.env.local` and fill `MOUSEGPT_API_KEY` for the default official OpenAI API. The same file also configures Stage 2 and 3 requests; see [configuration](../CONFIGURATION.md). Then run from `4_NLUI/`:

```bash
python agent.py
```

Open **<http://127.0.0.1:7860>**, choose a session, and use the frame slider or chat to explore its behavior. By default, conversation uses `gpt-4o`; semantic search and label assignment use `text-embedding-3-large`. Queries and relevant descriptions are sent to the configured provider and may incur usage charges.

| Explore | Example prompt | Method |
| --- | --- | --- |
| Inspect a frame | “Show the description of this frame.” | Retrieve the stored description |
| Find a behavior | “Find frames showing grooming.” | Rank descriptions by embedding similarity |
| Assign labels | “Classify the frames as grooming, walking, or resting.” | Match embeddings to user-defined label descriptions |
| Discover groups | “Cluster the behavior in this session.” | Traverse the stored hierarchy with language-model merge decisions |
| Inspect unusual frames | “Which frames are most unusual?” | Rank stored embeddings with IsolationForest |

Search, classification, clustering, and outlier tools export their results under `workspace/`. CSV results retain the HDF5 frame IDs so they can be linked back to session images. Conversation logs are stored under `workspace/user_logs/`.

## Use your own session

Add `SESSION_ID.h5` and a matching `SESSION_ID_imgs/` folder to `data/`. Preview filenames must use the stored frame ID, for example `00000012.jpg`.

| HDF5 table | Required contents |
| --- | --- |
| `descriptions` | Unique `id` values and their `description` records |
| `features_des` | Finite, nonzero description embeddings, indexed by the same IDs in the same row order |
| `linkage_matrix` | A hierarchy over those feature rows, required by the clustering tool |

Search and classification require embeddings compatible with `text-embedding-3-large`, including the same dimensions. The app validates description/feature alignment before loading a session and displays previews with matching stored records. See [Stage 3](../3_applications/readme.md) for description embedding and session packaging.

The distributed example contains 5,999 description/feature rows and 6,000 JPEGs. One additional image, `00037368.jpg`, has no stored feature row and is excluded from the interface. Frame IDs define correspondence; image-list positions do not.

## Interpret the outputs

Similarity search returns the closest descriptions to a query. Label assignment selects among the supplied label descriptions; its categories therefore depend on the labels you choose. IsolationForest identifies unusual embeddings within the loaded session, which need not represent a distinct or rare behavior in a broader population.

Clustering labels only groups with a confirmed merge. Frames that remain unresolved at a leaf or traversal depth limit receive `Unassigned` (`-1`), and remain included in exported counts and proportions. Language-model categories can vary between runs, so check them against the intended behavioral definitions before drawing quantitative conclusions.

## Runtime configuration

The interface listens on localhost and processes requests sequentially. Its Python REPL tool can execute code in the local process; use it in a trusted local session. Direct clustering and embedding requests use finite connection/read timeouts and at most three attempts.

LangSmith/LangChain tracing is disabled by default. To enable it, set `LANGCHAIN_TRACING_V2=true` and configure your tracing credentials before launch.

For new video and pose processing, start with [Stage 1](../1_keypoints/readme.md) and [Stage 2](../2_VLM/readme.md). Cohort-level phenotype analysis is available separately in [Stage 3](../3_applications/readme.md#phenotype-analysis).
