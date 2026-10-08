# Training a mouse pose model

This guide describes how to configure the bundled MMPose integration for your own annotated images. Install the environment described in the [Stage 1 README](../../readme.md) before proceeding. Run the commands below from `1_keypoints/`.

## Organize annotations

Use COCO-style keypoint annotations with three top-level arrays: `images`, `annotations`, and `categories`.

- Each image entry includes a unique `id`, `file_name`, `height`, and `width`.
- Each annotation includes `id`, `image_id`, `category_id`, `bbox`, `area`, `iscrowd`, `num_keypoints`, and `keypoints`.
- `bbox` uses `[x, y, width, height]` in image pixels.
- `keypoints` is a flattened list of `(x, y, v)` triplets. COCO visibility values are `0` for unlabelled, `1` for labelled but not visible, and `2` for labelled and visible.
- Categories define the keypoint names and skeleton in the same order used by the dataset metadata and model output.

For example:

```text
dataset/
├── annotations/
│   ├── train.json
│   └── val.json
└── images/
    ├── frame_00001.jpg
    └── frame_00002.jpg
```

Create separate training and validation splits at the appropriate experimental unit for your study. Keep annotation IDs unique within each split.

## Configure the dataset and model

Mouse dataset classes are in [mmpose/datasets/datasets/animal/](../../mmpose/datasets/datasets/animal/), with metadata in [configs/_base_/datasets/](../../configs/_base_/datasets/). The metadata defines keypoint order, left/right pairs, skeleton links, and evaluation parameters.

Choose a configuration under [configs/animal_2d_keypoint/](../../configs/animal_2d_keypoint/) that matches the model architecture and keypoint scheme. Copy it to a configuration file for your experiment, and set:

- The registered dataset type and corresponding metadata.
- `data_root`, annotation paths, and image prefixes for each split.
- The number of model output keypoints, codec settings, and evaluation configuration.
- Batch size, optimizer, schedule, augmentations, and any pretrained checkpoint.

For example, a dataset entry within a dataloader can use:

```python
dataset=dict(
    type='mouse_v6',
    data_root='/path/to/dataset',
    data_mode='topdown',
    ann_file='annotations/train.json',
    data_prefix=dict(img='images/'),
    pipeline=train_pipeline,
)
```

The `mouse_v6` dataset expects the keypoint scheme defined in [mouse_v6.py](../../configs/_base_/datasets/mouse_v6.py). Select a different registered dataset type when your annotations use a different scheme. Update validation and test loaders and evaluators consistently.

When combining datasets, use a `KeypointConverter` with an explicit mapping into a shared keypoint scheme. A mapping is specific to the source and target annotation definitions and must not be reused across incompatible schemes. See the bundled [mixed-dataset configurations](../../configs/animal_2d_keypoint/topdown_heatmap/mouse_v3/) for examples.

## Train and evaluate

Pass your configuration and output directory explicitly:

```bash
python tools/train.py \
  --config /path/to/experiment.py \
  --work-dir /path/to/training-output
```

Save the resolved configuration alongside the checkpoints. For evaluation, inspect the available arguments with `python tools/test.py --help` and use the same dataset definitions and keypoint ordering as training.

For video inference and multiview reconstruction, continue with the [Stage 1 inference instructions](../../readme.md). A detection checkpoint and its configuration are also required for top-down pose inference.
