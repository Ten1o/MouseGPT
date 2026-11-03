# Stage 3 · Behavior analysis

Turn MouseGPT descriptions into embeddings, behavior categories, and session summaries. This stage provides local anomaly ranking, coarse and fine behavior analysis, and a separate workflow for exploring treatment-related phenotypes.

Start with the prepared example session to explore existing descriptions, or generate a session HDF5 file from [Stage 2](../2_VLM/readme.md) outputs.

| Workflow | Input | Output | External service |
| --- | --- | --- | --- |
| Description embedding | Per-frame descriptions | Vectors with recorded model provenance | Configured `EMBEDDING` service |
| Session packaging | Descriptions, vectors, and their provenance | Aligned HDF5 tables; optional clustering hierarchy | None |
| Anomaly ranking | Session embeddings | Rows ranked by IsolationForest score | None |
| Coarse behavior analysis | Embeddings and description keywords | Behavior categories and session counts | Configured `ANALYSIS` service |
| Fine behavior analysis | Body-part descriptions and initial categories | Body-part labels and session counts | Configured `ANALYSIS` service |
| Phenotype analysis | Cohort feature CSV and labeled reference sessions | Treatment predictions and calculation reports | OpenAI Responses API with Code Interpreter |

## Installation

For description embedding and session packaging, use the root API environment from the [quick start](../README.md#quick-start-use-a-hosted-vlm). From the repository root:

```bash
python -m pip install -r requirements-api.txt
cd 3_applications
```

The remaining commands in this guide run from `3_applications/`. Full anomaly, UMAP, coarse/fine cohort, and report workflows need additional scientific packages. The historical `environment.yml` describes that analysis environment and includes Linux-specific builds:

```bash
conda env create -f environment.yml
conda activate mgpt
```

Use a separate environment for those specialist workflows. On another platform, resolve compatible packages and verify the requested analysis before using it; the exported environment is not a portable installer. The hosted quick start does not require those cohort dependencies.

API-backed workflows read the repository-root `.env.local` through the shared configuration loader. For the default official OpenAI API, copy `.env.example` there and fill `MOUSEGPT_API_KEY`; see [configuration](../CONFIGURATION.md) for optional role-specific services. They send selected descriptions or cohort data to the provider and may incur usage charges. Local anomaly ranking, session packaging, and dimensionality reduction do not require an API key.

## Try the prepared session

Download `nlui-demo` using the [example-data instructions](../examples/README.md), then rank its frames by embedding anomaly score:

```bash
python novel_behavior.py --h5 ../4_NLUI/data/240419_C11M4.h5
```

The command prints a zero-based HDF5 row number and an IsolationForest score for each row. Lower scores indicate more unusual embeddings. These row numbers are positions in the feature table, rather than camera frame IDs. To explore the ranked frames visually, use the [natural-language interface](../4_NLUI/readme.md).

The prepared session includes descriptions, embeddings, and image previews, so this example does not require raw videos or model inference.

## Build a session from your descriptions

Arrange the Stage 2 descriptions under a session directory:

```text
workspace/
└── SESSION_ID/
    ├── descriptions/
    │   ├── FRAME_ID.json
    │   └── ...
    └── feature/             # Created by the embedding script
```

Generate embeddings and package the session:

```bash
python embedding_request.py --data_folder workspace/SESSION_ID
python create_h5.py --workspace workspace --group_list SESSION_ID --with-linkage
```

The output is `workspace/SESSION_ID.h5`, containing `descriptions`, `features_des`, and (with `--with-linkage`) `linkage_matrix` tables. The model comes from the `EMBEDDING` configuration; the default is `text-embedding-3-large`. Descriptive fields are combined into one text input per frame, excluding `Keywords`.

Supported inputs include JSON description objects, JSON chat-completion responses, and TXT files containing plain text or JSON descriptions. Each filename stem is the frame ID, and each embedding is saved as `FRAME_ID_des.csv`. IDs must be unique and match across descriptions and embeddings. Vectors may have any consistent positive dimension and must be finite and nonzero. The exporter validates the complete session, including each vector's provenance sidecar, before atomically replacing an existing HDF5 file. Keep a separate workspace when you need to retain an earlier session. For structured analysis, retain the Stage 2 fields: coarse clustering uses `Keywords`, and fine analysis uses the selected body-part field.

The script writes provenance under `feature/.embedding-cache/`, recording the configured service/model, settings, input hash, and vector hash. Keep these files with the CSV vectors. Packaging verifies that all vectors come from the same configuration and still match their descriptions; it stores the identity in the HDF5 metadata. CSVs without valid metadata must be regenerated rather than assigned an assumed model name.

`--with-linkage` builds average linkage over cosine distances in the stored embedding space. It preserves the exact feature-row order and creates an empty four-column linkage table for a one-frame session. Distance computation uses quadratic memory, so begin with a small session and assess memory requirements before using it on a long recording. Omit this option when only search/classification is needed. This per-session hierarchy is distinct from the UMAP/cohort analysis described below.

Embedding requests use three attempts per file and eight concurrent workers by default. Adjust `--max_attempts`, `--max_workers`, and `--timeout` for your workload. Matching cached embeddings are reused; changing the description, embedding model, or vector file triggers regeneration. A failed file makes the command exit with an error. For a first run, use `--max_workers 1`.

From the repository root, validate and search the packaged session with:

```bash
python scripts/query_session.py \
  --session 3_applications/workspace/SESSION_ID.h5 --validate-only
python scripts/query_session.py \
  --session 3_applications/workspace/SESSION_ID.h5 --query "mouse walking" \
  --output 3_applications/workspace/search.csv
```

Search and classification check stored embedding identity against the current `EMBEDDING` role before requesting query vectors. Changing models or services requires regenerating and repackaging the session. For legacy HDF5 files without provenance, regenerate from the source descriptions; only the exact published prepared-session file is recognized as its documented original configuration. Use stable model IDs: metadata cannot detect a provider silently changing the model behind an unchanged deployment alias.

## Configure sessions and groups

Define your session groups in [config.py](config.py). The included `example` group selects `240419_C11M4`; add the relevant session IDs for your own cohort comparisons.

Analysis scripts read `WORKSPACE/MODEL_NAME/SESSION_ID.h5`, where `MODEL_NAME` is a directory label for a set of generated descriptions. Use a working copy because the coarse and fine analysis scripts add category and label tables to these HDF5 files:

```bash
mkdir -p workspace/example
cp ../4_NLUI/data/240419_C11M4.h5 workspace/example/
```

For a newly packaged session, place its HDF5 file in the corresponding model directory and select its group configuration.

## Coarse behavior analysis

First compute a two-dimensional UMAP projection and single-linkage hierarchy:

```bash
python dim_reduct_linkage_mat.py \
  --model example --group_config example --n_neigh 100 \
  --workspace workspace --output output
```

Then group and name behaviors using description keywords:

```bash
python llm_cluster.py \
  --model_name example --n_neigh 100 --min_dist 0.1 \
  --group_config example --downscale 1 \
  --npy_folder output --workspace workspace --img_workspace workspace \
  --ofolder output --date demo
```

Use the same model label, group configuration, neighbor count, and `min_dist` in both commands. Keep `--downscale 1` to preserve per-frame alignment. The `--img_workspace` argument identifies the prepared session workspace; this clustering workflow uses text keywords and does not send images to the model.

`--n_clusters` sets the initial cluster target (default: 400), and `--min_cluster_size` sets the minimum cluster size (default: 3). Choose these settings for your session size and behavioral resolution. Outputs include category and label tables in each HDF5 file and an Excel table of session-level counts in `output/`.

## Fine behavior analysis

Use the included body-part prompts and initial categories to assign more detailed labels. For head behavior:

```bash
python fine_behaviors.py \
  --output_json head_categories.json --category init_json/head_no_left_right.json \
  --workspace workspace --prompt prompt/head_prompt_no_left_right.txt \
  --part_name Head --model_name example --group_config example \
  --date demo --ofolder output
```

The script writes updated categories, per-frame HDF5 labels, and an Excel count table. Missing body-part descriptions and descriptions that remain unclassified receive label `-1`; assigned labels index the stored category table.

Both coarse and fine workflows accept `--max_attempts` (default: 3) for each category or merge response. Fine analysis also limits category updates with `--max_category_updates` (default: 100). Check generated categories against your behavioral definitions before using them in quantitative comparisons. Cohort estimates require the corresponding sessions, experimental groups, and analysis settings.

## Phenotype analysis

The phenotype workflow analyzes session-level behavior features using labeled reference sessions and a disjoint test set:

```bash
python behavior_phenotype_predict.py \
  --data_csv report/coarse2fine_internvl2-full-18000_medicine_all.csv \
  --groups_json report/cohorts.json \
  --output output/phenotype
```

See [Phenotype analysis inputs](report/README.md) for the CSV format, cohort configuration, and optional narrative context. The script uploads the selected CSV to the OpenAI Responses API with Code Interpreter and saves one prediction report per test session, together with an input manifest. Test labels are retained in the output for comparison and excluded from the prediction prompt.

The default model is `gpt-4o`. Use `--model` for another Responses model that supports Code Interpreter and `--max_wait` to set the request timeout in seconds (default: 600). An optional `--context_report PATH.txt` supplies narrative context; no report is loaded by default. This is an API-based analysis workflow, separate from the NLUI, rather than a distributed trained phenotype classifier. Its outputs should be assessed using the selected cohort design and held-out labels.
