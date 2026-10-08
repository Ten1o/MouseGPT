# Phenotype analysis inputs

This directory contains session-level behavior summaries and example cohort definitions for the [phenotype analysis workflow](../readme.md#phenotype-analysis).

| File | Contents |
| --- | --- |
| [coarse2fine_internvl2-full-18000_medicine_all.csv](coarse2fine_internvl2-full-18000_medicine_all.csv) | Behavior counts and ratios for individual sessions. `Group` identifies the session; feature names distinguish count and ratio measurements. |
| [cohorts.json](cohorts.json) | Six labeled reference sessions (`knowledge_group`) and 23 test sessions (`test_group`). |
| [report_medicine.txt](report_medicine.txt) | Optional narrative context describing behavioral comparisons and treatment-related interpretations. |

## Select a cohort

Each entry in `cohorts.json` is a `[session ID, treatment label]` pair. Replace these entries when analyzing another cohort. Reference and test sessions must be disjoint, and their IDs must correspond to the selected feature CSV.

The prediction workflow supplies reference labels to the model. Test labels are kept out of the prediction prompt and written alongside the returned predictions for evaluation. The CSV contains derived behavior features; it is not the raw video or pose dataset.

## Add narrative context

The workflow uses the CSV and cohort configuration by default. To include the narrative, add the following option when running `behavior_phenotype_predict.py` from `3_applications/`:

```bash
--context_report report/report_medicine.txt
```

Context is supplied separately from the numerical data and becomes part of the recorded analysis input. The narrative includes statistical statements, but the accompanying files do not include the raw observations and test specifications needed to reproduce those statements. Use the underlying experimental data and an explicit statistical analysis to establish group differences; model predictions and narrative interpretations serve different purposes.
