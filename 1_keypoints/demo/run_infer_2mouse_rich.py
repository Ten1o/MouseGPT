import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from glob import glob

det_model = os.environ['MOUSEGPT_DETECTOR_CONFIG']
det_weights = os.environ['MOUSEGPT_DETECTOR_WEIGHTS']
pose_model = os.environ['MOUSEGPT_POSE_CONFIG']
pose_weights = os.environ['MOUSEGPT_POSE_WEIGHTS']

nas_root = os.environ['MOUSEGPT_DATA_ROOT']
date = [value.strip() for value in os.environ['MOUSEGPT_DATES'].split(',') if value.strip()]
a = 0

out_root = os.environ['MOUSEGPT_OUTPUT_DIR']
os.makedirs(out_root, exist_ok=True)

for day in date:
    group_path = f"{nas_root}/{day}"
    day_out_path = f"{out_root}/{day}"
    os.makedirs(day_out_path, exist_ok=True)
    print(group_path)
    for group in sorted(glob(f"{group_path}/{os.environ['MOUSEGPT_GROUP_GLOB']}")):
        videos_path = f"{group}/videos"
        group_out_path = f"{out_root}/{day}/{group.split('/')[-1]}"
        os.makedirs(group_out_path, exist_ok=True)

        a = 1

        for video in sorted(glob(f"{videos_path}/*.mp4")):
            print(video)

            a += 1
            if a % 2 == 1:
                continue

            json_path = f'{group_out_path}/{video.split("/")[-1].split(".")[0]}.json'

            if os.path.exists(json_path):
                continue

            cmd = [
                sys.executable, str(Path(__file__).resolve().with_name('inferencer_demo.py')), video,
                "--pose2d", pose_model, "--pose2d-weights", pose_weights,
                "--det-model", det_model, "--det-weights", det_weights,
                "--vis-out-dir", group_out_path, "--pred-out-dir", group_out_path, "--draw-bbox",
                '--bbox-thr', '0.6', '--nms-thr', '0.6',
            ]
            print(cmd)
            subprocess.run(cmd, check=True)


