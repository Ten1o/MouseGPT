'''
1. Crop images by its bbox
2. Output velocity.json
'''

import json
import os
import numpy as np
from scipy.signal import savgol_filter
from PIL import Image
from tqdm import tqdm
import argparse

def filter(data_3d):
    window_length = 15
    poly_order = 3
    if window_length % 2 == 0:
        window_length += 1
    filtered_data_3d = np.empty_like(data_3d)
    for dim in range(3):
        for point in range(data_3d.shape[1]):
            filtered_data_3d[:, point, dim] = savgol_filter(data_3d[:, point, dim], window_length, poly_order)
    return filtered_data_3d

def keypoint_velocity(kpts, frame_index, fps):
    """Return cm/s from metre coordinates at the original video frame rate.

    Interior frames use a centered difference; the first and last frames use
    forward and backward one-sided differences, respectively.
    """
    left = max(0, frame_index - 1)
    right = min(len(kpts) - 1, frame_index + 1)
    return (kpts[right] - kpts[left]) * (fps / (right - left)) * 100

def process_data(data_folder, json_folder, fps):
    origin_folders = os.path.join(data_folder, "images")
    output_folders = os.path.join(data_folder, "pre_images")
    motion_file = os.path.join(data_folder, "velocity.json")

    kpts = np.load(os.path.join(json_folder, 'tri_filter.npy'))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and greater than zero")
    if kpts.ndim != 3 or kpts.shape[0] < 2 or kpts.shape[1] < 22 or kpts.shape[2] != 3:
        raise ValueError("tri_filter.npy must have shape (frames >= 2, keypoints >= 22, 3)")
    if not np.isfinite(kpts).all():
        raise ValueError("tri_filter.npy contains non-finite coordinates")

    for data_id in [3, 5]:
        origin_folder = os.path.join(origin_folders, f"data{data_id}")
        output_folder = os.path.join(output_folders, f"data{data_id}")
        os.makedirs(output_folder, exist_ok=True)

        json_file = os.path.join(json_folder, f"data{data_id}.json")
        with open(json_file, 'r') as f:
            bboxes = json.load(f)

        motion = {}
        filenames = [name for name in sorted(os.listdir(origin_folder))
                     if name.lower().endswith((".jpg", ".png"))]
        for filename in filenames:
            frame_index = int(os.path.splitext(filename)[0])
            if not 0 <= frame_index < len(kpts):
                raise ValueError(f"Image frame index {frame_index} is outside tri_filter.npy")
            if frame_index >= len(bboxes):
                raise ValueError(f"No bounding box for frame {frame_index} in {json_file}")
        for filename in tqdm(filenames, desc=f"Processing data{data_id}"):
            if data_id == 3:
                kpts_idx = int(os.path.basename(filename).split(".")[0])

                speed = keypoint_velocity(kpts, kpts_idx, fps)
                base = speed[4].copy()
                speed = speed - base
                motion[kpts_idx] = {
                    "base": float(np.linalg.norm(base)),
                    "head": float(np.linalg.norm(speed[2])),
                    "forepawL": float(np.linalg.norm(speed[10])),
                    "forepawR": float(np.linalg.norm(speed[14])),
                    "hindpawL": float(np.linalg.norm(speed[18])),
                    "hindpawR": float(np.linalg.norm(speed[21])),
                }

            # Open the corresponding camera frame.
            img_path = os.path.join(origin_folder, filename)
            with Image.open(img_path) as source_image:
                img = source_image.copy()

            # Expand the detected bounding box to a square.
            bbox_idx = int(os.path.basename(filename).split(".")[0])
            bbox = bboxes[bbox_idx]["instances"][0]["bbox"][0]
            left, upper, right, lower = bbox

            width = right - left
            height = lower - upper

            if width > height:
                diff = width - height
                upper -= int(diff // 2 * 1.1)
                lower += int((diff - diff // 2) * 1.1)
                left -= int(width * 0.05)
                right += int(width * 0.05)
            else:
                diff = height - width
                left -= int(diff // 2 * 1.1)
                right += int((diff - diff // 2) * 1.1)
                upper -= int(height * 0.05)
                lower += int(height * 0.05)

            square_img = img.crop((left, upper, right, lower))

            resized_img = square_img.resize((512, 512))

            resized_img.save(os.path.join(output_folder, f"{filename}"))
        if data_id == 3:
            with open(motion_file, 'w') as file:
                json.dump(motion, file)

def main():
    parser = argparse.ArgumentParser(description="Process images and keypoints from video data.")
    parser.add_argument("--data_folder", required=True, type=str, help="Base folder for data processing.")
    parser.add_argument("--json_folder", required=True, type=str, help="Folder containing keypoints in .npy format.")
    parser.add_argument("--fps", required=True, type=float, help="Fps of the video for velocity calculation.")

    args = parser.parse_args()
    process_data(args.data_folder, args.json_folder, args.fps)

if __name__ == "__main__":
    main()
