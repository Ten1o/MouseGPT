# Frame-description protocol

The paper analysis treats 435 manually selected frames as one benchmark. Each
frame combines two synchronized images and the permitted kinematic context;
historical acquisition tags are not difficulty labels. Frame identity combines
recording identity and frame number. A recording identifier is not proof of an
independent animal.

The six requested output fields are `Overall`, `Head`, `Limb`, `Torso`, `Others`,
and `Keywords`. Candidate models receive the bound sensory input and task
instructions, without reference answers, prior candidates, or ratings. Approved
references contain atomic facts and their permitted evidence. Unsupported or
uncertain facts cannot become required recall targets. Drafting/checking is
separate from explicit human reference approval.

## Graded factual score

The exact executable anchors and judgment contract are in
[`protocol/rubric.json`](protocol/rubric.json). Its canonical content hash is
`eee578b5bd8eda7d802c1543ada34712e08176b0b77b7a9a4dedb7ac1bcd3142`.
The rubric is fixed for score computation. Its grading scales and weights have
not been calibrated against a held-out human evaluation.

Each distinct candidate factual claim receives a correctness credit and each
required reference fact a separate coverage credit, from `0, 0.25, 0.5, 0.75, 1`.
Weights per factual point are Overall 2; Head, Limb and Torso 1; Others 0.5.
Repeated Keywords propositions do not earn duplicate credit. Novel keyword
claims use their semantic field; legacy reference fallback weights remain
explicit in the scorer.

```text
P = sum(claim_weight * correctness) / sum(claim_weight)
R = sum(required_fact_weight * coverage) / sum(required_fact_weight)
F = 2 * P * R / (P + R)
index = 100 * F * (0.80 + 0.10 * structure + 0.10 * instruction)
```

The index is calculated per frame before averaging. This custom graded factual
F-score is not classification F1, and the composite index is not accuracy.
Structure and instruction grades are separate anchored judgments. Error types
and checklist findings are diagnostics, not additional factual deductions.
Zero candidate claims give zero precision. No required reference facts makes
recall/F unscorable. Positive coverage requires a linked claim with positive
correctness. Equivalent duplicated claims count once.

## Comparisons and uncertainty

The saved paper analysis includes 14 conditions over eight model identities,
with complete paired coverage of the 435 frames (6,090 assessments). Inputs,
references, rubric, and judge configuration are fixed. Candidate settings differ
in reasoning effort and output limits. The frozen saved judge is recorded in
`figure3/methodology.json`; choosing a current similarly named deployment does
not recreate a historical execution.

Paper mean intervals use 10,000 recording-cluster bootstrap resamples, seed
20261008 (NumPy PCG64). Sample 25 recording identifiers with replacement; all
frames and conditions within a sampled recording travel together. A replicate
is the sum of sampled scores divided by the sampled frame count. Take the 2.5th
and 97.5th percentiles using linear quantiles. These are conditional intervals
for the selected recordings and saved judgments, not animal-population,
provider-repeatability, or judge-repeatability intervals. The workbench's older
pairwise diagnostic bootstrap uses its own documented settings; it must not be
substituted for the paper's bootstrap.

The descriptive lowest-scoring subset averages effort conditions within model
identity, then weights the eight identities equally. The lowest `ceil(0.10*435)`
= 44 frames are shared by all 14 plotted conditions. Comparing that subset with
all 435 frames uses overlapping, outcome-selected sets. No subset CI or
independent validation claim is made. Leave-one-identity-out sensitivity is
exported by `figure3/subset_analysis.py`.

## Limits and data provenance

Manual frame selection does not establish representative sampling or training
independence. Human-approved references do not establish human–judge agreement.
No verified held-out human calibration validates the custom grades, weights,
or index factors. Historical model labels and archived human ratings must retain
their own provenance; they are not directly interchangeable with the fresh
controlled benchmark. See the saved methodology record for additional limits.

The figure input manifest records filenames, byte sizes, and SHA-256 hashes for
the required source tables and example images. These files must be supplied
separately from the code. Data and third-party model outputs remain subject to
their respective terms of use.
