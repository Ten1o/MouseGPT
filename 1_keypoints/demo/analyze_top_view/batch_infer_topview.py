"""Run top-view inference using explicit data, checkpoint and output paths."""
import os
from glob import glob
from pathlib import Path
import subprocess
import sys

video_glob = os.environ['MOUSEGPT_VIDEO_GLOB']
pose_model = os.environ['MOUSEGPT_POSE_CONFIG']
pose_weights = os.environ['MOUSEGPT_POSE_WEIGHTS']
det_model = os.environ['MOUSEGPT_DETECTOR_CONFIG']
det_weights = os.environ['MOUSEGPT_DETECTOR_WEIGHTS']
out_folder = os.environ['MOUSEGPT_OUTPUT_DIR']
os.makedirs(out_folder, exist_ok=True)

for video in sorted(glob(video_glob)):
    command = [
        sys.executable, str(Path(__file__).resolve().parents[1] / 'inferencer_demo.py'), video,
        '--pose2d', pose_model, '--pose2d-weights', pose_weights,
        '--det-model', det_model, '--det-weights', det_weights,
        '--vis-out-dir', out_folder, '--pred-out-dir', out_folder, '--draw-bbox',
    ]
    subprocess.run(command, check=True)
