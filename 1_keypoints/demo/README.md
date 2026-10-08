# Pose analysis and visualization tools

These utilities extend the [Stage 1 workflow](../readme.md) with batch inference, keypoint visualization, calibration analysis, and trajectory plots. Use the main Stage 1 guide for single-video inference and eight-camera reconstruction.

## Configure your inputs

Each utility reads its inputs from environment variables or explicit command-line arguments. Set only the variables required by the selected utility; the [reference table](#helper-reference) lists them. Notebooks begin with a check of their required configuration. Input paths should identify existing files or directories, and output paths should point to your desired destination. Variables ending in `_GLOB` accept Python glob patterns.

| Variable | Meaning |
| --- | --- |
| `MOUSEGPT_DATA_ROOT` | Dataset root used by the selected helper; check its relative session layout. |
| `MOUSEGPT_DATE`, `MOUSEGPT_DATES` | One date directory, or a comma-separated list of date directories. |
| `MOUSEGPT_DATE_GLOB` | Pattern selecting date directories that contain session directories. |
| `MOUSEGPT_GROUP_GLOB` | Session-directory pattern within each date directory. |
| `MOUSEGPT_SESSION_DIR` | Directory containing one session's data. |
| `MOUSEGPT_VIDEO`, `MOUSEGPT_VIDEO_DIR`, `MOUSEGPT_VIDEO_GLOB` | Video file, video directory, or video-file pattern. |
| `MOUSEGPT_POSE_CONFIG`, `MOUSEGPT_POSE_WEIGHTS` | Matching pose configuration and checkpoint. |
| `MOUSEGPT_DETECTOR_CONFIG`, `MOUSEGPT_DETECTOR_WEIGHTS` | Matching detector configuration and checkpoint. |
| `MOUSEGPT_KEYPOINTS_JSON`, `MOUSEGPT_KEYPOINTS_NPY` | Keypoint JSON file or NumPy array. |
| `MOUSEGPT_KEYPOINTS_DIR`, `MOUSEGPT_HDF5_DIR` | Directories containing session NumPy arrays or HDF5 files. |
| `MOUSEGPT_CALIBRATION_MAT`, `MOUSEGPT_CALIBRATION_DIR`, `MOUSEGPT_CALIBRATION_GLOB` | Calibration MAT file, directory, or file pattern. |
| `MOUSEGPT_OUTPUT_DIR`, `MOUSEGPT_OUTPUT_VIDEO` | Output directory or video file. |

Frame rate, spatial units, body-point definitions, chamber geometry, filter settings, and cohort selections are specified in the analysis scripts and notebooks. Set these to match your recordings before calculating measurements or generating figures.

## Batch inference

Run from `1_keypoints/` in the environment described in the [Stage 1 installation guide](../readme.md#environment):

```bash
export MOUSEGPT_VIDEO_DIR=/path/to/videos
export MOUSEGPT_POSE_CONFIG=/path/to/pose-config.py
export MOUSEGPT_POSE_WEIGHTS=/path/to/pose-checkpoint.pth
export MOUSEGPT_DETECTOR_CONFIG=/path/to/detection-config.py
export MOUSEGPT_DETECTOR_WEIGHTS=/path/to/detection-checkpoint.pth
export MOUSEGPT_OUTPUT_DIR=/path/to/results

python demo/batch_infer.py
```

`batch_infer.py` processes each entry in the supplied directory, so use a directory containing only the intended videos. `analyze_top_view/batch_infer_topview.py` accepts `MOUSEGPT_VIDEO_GLOB` instead. Both stop the batch if an inference command fails.

For recordings organized as `<root>/<date>/<session>/videos`, use `run_infer.py` or `run_infer_2mouse_rich.py` with `MOUSEGPT_DATA_ROOT`, `MOUSEGPT_DATES`, and `MOUSEGPT_GROUP_GLOB`. Results are written to `<output>/<date>/<session>`.

These two utilities process alternating videos: `run_infer.py` uses a counter across sessions; `run_infer_2mouse_rich.py` selects the first and every subsequent second video within each session. Choose `batch_infer.py` when every video should be processed.

`check/batch_vis_write.py` uses `MOUSEGPT_CALIBRATION_DIR`, `MOUSEGPT_DATA_ROOT`, and `MOUSEGPT_OUTPUT_DIR`. Its calibration filenames determine date, group, and demo subdirectories under the dataset root.

## Trajectory plots

`trajectory/labeled_point_map.py` plots trajectories colored by behavior category. It reads session HDF5 files and matching keypoint arrays:

```bash
python demo/trajectory/labeled_point_map.py \
  --model internvl2-full-18000 \
  --h5-folder /path/to/session-hdf5 \
  --npy-folder /path/to/session-keypoints \
  --output_path /path/to/trajectory-figures
```

The HDF5 folder must contain session files for the selected model, and NumPy filenames must match those session IDs. The script uses a predefined cohort mapping and four behavior categories; adapt these to the collection being analyzed.

## Helper reference

| Helper | Required environment variables |
| --- | --- |
| `analyze_top_view/batch_cal_body_angle_filter_17.py` | `MOUSEGPT_DATA_ROOT`, `MOUSEGPT_DATES`, `MOUSEGPT_GROUP_GLOB` |
| `analyze_top_view/batch_infer_topview.py` | `MOUSEGPT_DETECTOR_CONFIG`, `MOUSEGPT_DETECTOR_WEIGHTS`, `MOUSEGPT_OUTPUT_DIR`, `MOUSEGPT_POSE_CONFIG`, `MOUSEGPT_POSE_WEIGHTS`, `MOUSEGPT_VIDEO_GLOB` |
| `analyze_top_view/get_center_region.py` | `MOUSEGPT_VIDEO` |
| `analyze_top_view/statistic_center_region_time.py` | `MOUSEGPT_KEYPOINTS_JSON` |
| `analyze_top_view/statistic_center_region_time_16.py` | `MOUSEGPT_DATA_ROOT`, `MOUSEGPT_DATE`, `MOUSEGPT_GROUP_GLOB` |
| `batch_infer.py` | `MOUSEGPT_DETECTOR_CONFIG`, `MOUSEGPT_DETECTOR_WEIGHTS`, `MOUSEGPT_OUTPUT_DIR`, `MOUSEGPT_POSE_CONFIG`, `MOUSEGPT_POSE_WEIGHTS`, `MOUSEGPT_VIDEO_DIR` |
| `batch_vis.py` | `MOUSEGPT_DATE_GLOB` |
| `batch_vis_one_frame.py` | `MOUSEGPT_KEYPOINTS_NPY` |
| `check/batch_vis_write.py` | `MOUSEGPT_CALIBRATION_DIR`, `MOUSEGPT_DATA_ROOT`, `MOUSEGPT_OUTPUT_DIR` |
| `draw_2d_kpts.py` | `MOUSEGPT_KEYPOINTS_JSON`, `MOUSEGPT_VIDEO` |
| `find_lim.py` | `MOUSEGPT_DATE_GLOB` |
| `multi_vis.py` | `MOUSEGPT_DATE_GLOB` |
| `run_infer.py` | `MOUSEGPT_DATA_ROOT`, `MOUSEGPT_DATES`, `MOUSEGPT_DETECTOR_CONFIG`, `MOUSEGPT_DETECTOR_WEIGHTS`, `MOUSEGPT_GROUP_GLOB`, `MOUSEGPT_OUTPUT_DIR`, `MOUSEGPT_POSE_CONFIG`, `MOUSEGPT_POSE_WEIGHTS` |
| `run_infer_2mouse_rich.py` | `MOUSEGPT_DATA_ROOT`, `MOUSEGPT_DATES`, `MOUSEGPT_DETECTOR_CONFIG`, `MOUSEGPT_DETECTOR_WEIGHTS`, `MOUSEGPT_GROUP_GLOB`, `MOUSEGPT_OUTPUT_DIR`, `MOUSEGPT_POSE_CONFIG`, `MOUSEGPT_POSE_WEIGHTS` |
| `trajectory/gather_npy.py` | `MOUSEGPT_DATA_ROOT`, `MOUSEGPT_KEYPOINTS_DIR` |
| `triangulate/min_tri_mse.py` | `MOUSEGPT_CALIBRATION_GLOB`, `MOUSEGPT_DATA_ROOT` |
| `triangulate/min_tri_mse_batch_tri.py` | `MOUSEGPT_CALIBRATION_GLOB`, `MOUSEGPT_DATA_ROOT` |
| `triangulate/multi_mouse_tri.py` | `MOUSEGPT_CALIBRATION_MAT`, `MOUSEGPT_SESSION_DIR` |
| `triangulate/statistic_zero.py` | `MOUSEGPT_SESSION_DIR` |
| `velocity/batch_cal.py` | `MOUSEGPT_DATA_ROOT`, `MOUSEGPT_DATES` |
| `velocity/multi_vis.py` | `MOUSEGPT_DATE_GLOB` |
