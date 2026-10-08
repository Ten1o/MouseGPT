# Stage 1 · Mouse pose estimation and 3D reconstruction

This stage detects mice in synchronized camera recordings, estimates 2D keypoints, and reconstructs 3D trajectories. It includes MouseGPT's mouse-specific configurations and integrations with MMPose and MMYOLO.

**Licensing:** MMYOLO and the additional detector configurations listed in the [GPL scope](../LICENSES/GPL-SCOPE.md) are distributed under GPLv3, including MouseGPT modifications to those components. MMPose-derived source retains its Apache-2.0 terms and notices. See the [repository license](../LICENSE) for the scope of CC BY-NC-ND 4.0 on other MouseGPT-authored material.

**Inputs:** synchronized videos, matching detector and pose checkpoints, and camera calibration.

**Outputs:** per-camera keypoint predictions and a 3D keypoint array for downstream kinematic analysis.

Pretrained checkpoints will be distributed through [MouseGPT on Hugging Face](https://huggingface.co/Ten1o/MouseGPT); checkpoint uploads are forthcoming. The [example dataset](https://huggingface.co/datasets/Ten1o/MouseGPT-examples) provides inputs for later stages. Raw multi-camera recordings, calibration files, and pose training annotations are separate inputs to this stage.

## Environment

Use a dedicated Linux environment with CUDA-compatible PyTorch. Install MMCV with compiled operators, MMEngine, and MMDetection versions compatible with your PyTorch and CUDA installation and the following package constraints:

| Dependency | Version range |
| --- | --- |
| MMCV | `>=2.0.0,<2.1.0` |
| MMEngine | `>=0.7.1,<1.0.0` |
| MMDetection | `>=3.0.0,<3.1.0` |

Then install the bundled source trees from the repository root. These contain the MouseGPT integrations and report MMPose `1.0.0` and MMYOLO `0.6.0`.

```bash
python -m pip install -e ./1_keypoints/mmyolo
python -m pip install -e ./1_keypoints
```

See the [MMPose documentation](mmpose_README.md), [package requirements](requirements/), and [MMYOLO installation guide](mmyolo/docs/en/get_started/installation.md) for installation details. Triangulation also requires EasyMocap, NumPy, SciPy, pandas, OpenCV, and tqdm. Use an EasyMocap installation that provides the reconstruction and triangulator functions imported by [triangulation.py](demo/triangulation.py); this repository does not pin an EasyMocap version.

## 1. Estimate 2D keypoints

Run the following command from `1_keypoints/`, replacing the paths with the configurations and checkpoints for your models:

```bash
python demo/inferencer_demo.py /path/to/data1.mp4 \
  --pose2d /path/to/pose-config.py \
  --pose2d-weights /path/to/pose-checkpoint.pth \
  --det-model /path/to/detection-config.py \
  --det-weights /path/to/detection-checkpoint.pth \
  --vis-out-dir /path/to/results \
  --pred-out-dir /path/to/results \
  --draw-bbox
```

Repeat for each synchronized view, preserving frame order across cameras. Each configuration must match its checkpoint and keypoint definitions. For training on another collection, configure dataset paths, splits, and keypoint definitions using the [pose training guide](tutorials/mmpose.md/mmpose.md).

## 2. Reconstruct 3D keypoints

The reconstruction entry point uses **eight cameras and 24 keypoints**, selecting the first detected animal instance in each frame. Prepare:

| Input | Required format |
| --- | --- |
| 2D predictions | `data1.json` through `data8.json`, with matching frame order and counts across cameras. |
| Camera calibration | Eight MAT files in a dedicated directory, sorted in the same order as cameras 1–8. Each file must contain `K`, `r`, `t`, `RDistort`, and `TDistort` in the layout read by the script. |

From `1_keypoints/`:

```bash
python demo/triangulation.py \
  --cali_folder /path/to/calibration \
  --demo_folder /path/to/2d-predictions
```

The script writes `tri_new.npy` into the predictions directory. Its dimensions are **frame × keypoint × xyz**, and its spatial units follow the camera calibration. Frames for which reconstruction fails may contain `NaN`; handle these before passing trajectories to downstream preprocessing. The aliases `--cali_foler` and `--demo_foler` are also supported.

## 3. Prepare kinematic inputs for Stage 2

[Stage 2 preprocessing](../2_VLM/readme.md#prepare-images-and-kinematic-prompts) combines image crops with velocities computed from filtered 3D keypoints. It expects finite coordinates in **metres**, one row per original video frame, and the original video frame rate. Filtering and any unit conversion must be applied to the reconstructed trajectories before this step.

The VLM examples already contain kinematic prompts, and the behavior-exploration example includes precomputed descriptions and features. You can therefore try those workflows without first running pose estimation.

For batch inference, calibration analysis, trajectory plots, and notebook configuration, see the [additional tools guide](demo/README.md).
