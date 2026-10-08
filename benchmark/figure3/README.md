# Extended Data Figure 3

Reproduce the six-panel MouseGPT benchmark figure from saved score tables and
paired images. The figure covers benchmark design, behavioral examples, the
benchmark index, rubric-dimension comparisons including GPT-4o, overall results,
and performance on a shared lowest-scoring subset. Numerical comparisons use
dot plots and intervals.

## Required data

[expected_inputs.json](expected_inputs.json) specifies the filenames, sizes,
and SHA-256 hashes of all required inputs: 6,090 per-frame assessments, supporting
tables, four images, and an image manifest. These inputs must be supplied
separately from the code and are not included in the VLM or NLUI example bundles.

Place the files under `data/benchmark/figure3/`, preserving the relative paths
listed in the manifest:

```text
data/benchmark/figure3/
├── example_manifest.json
├── examples/
│   └── *.jpg
└── source_data/
    └── *.csv
```

## Render the figure

Use Python 3.12 and install Poppler so that `pdftoppm` is on `PATH`. From the
repository root:

```sh
python -m pip install -r benchmark/figure3/requirements.txt
python benchmark/figure3/verify_inputs.py --data data/benchmark/figure3
python benchmark/figure3/plot_figure.py
```

Artwork is written to `outputs/benchmark-figure3/`. The renderer verifies the
input hashes before reading scores or images. To use another location, set
`MOUSEGPT_FIGURE_DATA` for inputs or `MOUSEGPT_FIGURE_OUTPUT` for generated files.

| Output | Format |
| --- | --- |
| Individual panels | Editable PDF and SVG, with PNG previews. |
| Composite figure | 180 × 156 mm PDF and SVG. |
| Raster exports | 300 dpi PNG and 600 dpi LZW-compressed TIFF. |
| Layout preview | `composition.html`, with print-size layout and adjustable zoom. |

To reassemble previously generated panels, run `plot_figure.py --compose-only`.
The panel files must already exist in the selected output directory.

### Typography

Arial is resolved through the local font installation. To reproduce the paper
typography exactly, set `MOUSEGPT_FONT_REGULAR` and `MOUSEGPT_FONT_BOLD` to your
licensed Arial font files. If Arial is unavailable, the code warns and uses
DejaVu Sans. Font substitution can change line breaks and label placement.
Arial font files are not bundled with the code.

## Recompute the subset comparison

[subset_analysis.py](subset_analysis.py) reproduces the shared lowest-scoring
44-frame analysis from the saved 435-frame, 14-condition results. It verifies
complete coverage, averages reasoning-effort conditions within each model
identity, and then weights the eight identities equally. The outputs include
the full frame ranking and leave-one-identity-out sensitivity analyses.

Copy the three source tables into a separate output directory before running:

```sh
mkdir -p outputs/benchmark-analysis/source_data
cp data/benchmark/figure3/source_data/per_frame_scores.csv \
   data/benchmark/figure3/source_data/frame_factualF_difficulty.csv \
   data/benchmark/figure3/source_data/model_summary.csv \
   outputs/benchmark-analysis/source_data/
python benchmark/figure3/subset_analysis.py --output outputs/benchmark-analysis
```

The script writes derived tables alongside the copied inputs. The optional
`--independent-audit PATH` argument compares these outputs with separately
computed reference tables.

## Recompute scores from source records

[analyze_saved_scores.py](analyze_saved_scores.py) implements score calculation
and recording-cluster bootstrap directly from the saved evaluation records.
This route requires the source SQLite database, bound image objects, and
execution records in the directory structure expected by the script; the CSV
exports alone are insufficient. Supply both locations explicitly:

```sh
python benchmark/figure3/analyze_saved_scores.py \
  --project /path/to/source-project \
  --output outputs/benchmark-source-analysis
```

The script opens SQLite read-only, verifies the saved input and scoring
bindings, and makes no model-provider calls. Rendering the figure from exported
tables does not require this database.

## Interpretation

The benchmark treats all 435 frames as one collection. The paired example images
show a mouse lying belly-down and a hopping posture. Lowest-scoring membership
is determined from scores across model identities, rather than acquisition
tags; it is a descriptive subset rather than an independent validation set.

The overall intervals use 10,000 recording-cluster bootstrap resamples of
25 recordings, keeping paired model conditions together. They describe
selected-set sensitivity to recording composition with fixed predictions and
judgments. They do not quantify animal-population uncertainty or repeated model
and judge execution. The custom factual score and composite index also require
their own calibration. See [methodology.json](methodology.json) for exact
parameters and [the protocol](../PROTOCOL.md) for scoring definitions and limits.
