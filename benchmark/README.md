# Frame-description benchmark

Evaluate structured descriptions of mouse behavior using synchronized images,
kinematic context, and a fixed factual scoring rubric. This directory provides
the local evaluation interface, command-line workflow, scoring implementation,
and analysis code for Extended Data Figure 3.

The paper benchmark comprises **435 frames from 25 recordings**, evaluated under
14 conditions across eight model identities. All conditions share the same
frame collection. Acquisition tags record data provenance and do not define
difficulty groups.

| Component | Purpose |
| --- | --- |
| [Protocol](PROTOCOL.md) | Input contract, factual scoring, comparison design, and interpretation limits. |
| [Rubric](protocol/rubric.json) | Machine-readable grading anchors and weights. |
| `app.py` and `workflow.py` | Manage frame inputs, references, predictions, and judgments. |
| [Figure 3 pipeline](figure3/README.md) | Reproduce the figure and supporting analyses from saved scores. |

## Data requirements

The code is distributed separately from benchmark images, reference records,
and exported scores. These evaluation inputs are not part of the VLM or NLUI
example bundles. A new local installation starts with an empty catalog; import
your own evaluation data to use the workbench. Reproducing the paper figure
requires the specific inputs listed in
[figure3/expected_inputs.json](figure3/expected_inputs.json).

## Start the local workbench

Use Python 3.12 or newer. The workbench and its Python tests use only the standard
library. From the repository root:

```sh
python benchmark/app.py --data data/benchmark-workbench
```

Open <http://127.0.0.1:8765>. Imported frames, reference versions, predictions,
and judgments are stored in the directory supplied with `--data`.

Use the command-line workflow to inspect state and manage evaluation batches:

```sh
python benchmark/workflow.py --help
python benchmark/workflow.py --data data/benchmark-workbench status
```

Opening the interface or creating a batch does not initiate model inference.
API execution requires `run BATCH_ID --execute`, a configured endpoint and model
identity, and credentials supplied through environment variables.

## Evaluation design

Descriptions have six fields: `Overall`, `Head`, `Limb`, `Torso`, `Others`, and
`Keywords`. Human-approved reference facts define the recall targets; candidate
claims receive graded correctness scores. The primary factual score combines
weighted precision and recall. A secondary index incorporates structural and
instruction adherence. These are custom graded measures, rather than
classification F1 or accuracy; the exact definitions and calibration limits are
specified in the [protocol](PROTOCOL.md).

The workflow records hashes for inputs, references, predictions, rubrics, and
judge configurations. Scoring requires matching input and reference versions.
Candidate-model and judge settings are recorded separately for each run, and
imported historical outputs retain their original provenance.

The saved analysis uses recording-cluster bootstrap intervals and a descriptive,
shared lowest-scoring subset. Frames were manually selected, and recording IDs
do not establish independent animals. See the
[saved methodology](figure3/methodology.json) for resampling settings and the
limits of these comparisons.

## Offline verification

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=benchmark python -m unittest discover -s benchmark/tests -v
node --test benchmark/tests/*.cjs
```

Tests use synthetic fixtures and mocked provider transports. HTTP integration
tests bind the loopback interface and do not contact a model provider. Scientific
interpretation and calibration requirements are described in the protocol.

See the repository [license](../LICENSE) for terms of use.
