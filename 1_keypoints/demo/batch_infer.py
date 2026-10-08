import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from glob import glob

video_path = os.environ['MOUSEGPT_VIDEO_DIR']
det_model = os.environ['MOUSEGPT_DETECTOR_CONFIG']
det_weights = os.environ['MOUSEGPT_DETECTOR_WEIGHTS']
out_floder = os.environ['MOUSEGPT_OUTPUT_DIR']

os.makedirs(out_floder, exist_ok=True)

for video in sorted(glob(f'{video_path}/*')):
    # out_floder = f"./pose_res/rat_v5_{m}"
    pose_model = os.environ['MOUSEGPT_POSE_CONFIG']
    pose_weights = os.environ['MOUSEGPT_POSE_WEIGHTS']

    cmd = [
        sys.executable, str(Path(__file__).resolve().with_name('inferencer_demo.py')), video,
        "--pose2d", pose_model, "--pose2d-weights", pose_weights,
        "--det-model", det_model, "--det-weights", det_weights,
        "--vis-out-dir", out_floder, "--pred-out-dir", out_floder, "--draw-bbox",
    ]
    subprocess.run(cmd, check=True)
