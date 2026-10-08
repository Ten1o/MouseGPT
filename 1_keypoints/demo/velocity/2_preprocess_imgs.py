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

def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_angle = dot_product / magnitude_product
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle) - 90

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

def process_data(data_folder, json_folder):
    origin_folders = os.path.join(data_folder, "images")
    output_folders = os.path.join(data_folder, "pre_images")
    motion_file = os.path.join(data_folder, "velocity.json")
    
    kpts = np.load(os.path.join(json_folder, 'tri_filter_limbs.npy'))
    # kpts = filter(kpts)

    for data_id in [3, 5]:
        origin_folder = os.path.join(origin_folders, f"data{data_id}")
        output_folder = os.path.join(output_folders, f"data{data_id}")
        os.makedirs(output_folder, exist_ok=True)

        json_file = os.path.join(json_folder, f"data{data_id}.json")
        with open(json_file, 'r') as f:
            bboxes = json.load(f)

        motion = {}
        for i, filename in enumerate(tqdm(sorted(os.listdir(origin_folder)), desc=f"Processing data{data_id}")):
            if data_id == 3:
                kpts_idx = int(os.path.basename(filename).split(".")[0])
                
                if kpts_idx < 1:
                    motion[kpts_idx] = {
                        
                            "base": 0,
                            "head": 0, 
                            "forepawL": 0,
                            "forepawR": 0,
                            "hindpawL": 0,
                            "hindpawR": 0,
                        }
            
                else:
                    # cal vec
                    cur_kpts = kpts[kpts_idx]
                    fore_vec = cur_kpts[3] - cur_kpts[4]
                    base_vec = cur_kpts[4] - cur_kpts[5]
                    hind_vec = cur_kpts[6] - cur_kpts[5]

                    speed = (kpts[kpts_idx] - kpts[kpts_idx - 1]) * 60 * 100
                    base = speed[4].copy()
                    
                    speed = speed - base
                    motion[kpts_idx] = {
                            "base": np.linalg.norm(base) * np.sign(angle_between_vectors(base_vec, base)),
                            "head": np.linalg.norm(speed[2]), 
                            "forepawL": np.linalg.norm(speed[10]) * np.sign(angle_between_vectors(fore_vec, speed[10])), 
                            "forepawR": np.linalg.norm(speed[14]) * np.sign(angle_between_vectors(fore_vec, speed[14])), 
                            "hindpawL": np.linalg.norm(speed[18]) * np.sign(angle_between_vectors(hind_vec, speed[18])), 
                            "hindpawR": np.linalg.norm(speed[21]) * np.sign(angle_between_vectors(hind_vec, speed[21])), 
                        }
                    
            if data_id == 3 or data_id == 5:
                if filename.endswith(".jpg") or filename.endswith(".png"):

                        img_path = os.path.join(origin_folder, filename)
                        img = Image.open(img_path)


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
    parser.add_argument("--data_folder", type=str, help="Base folder for data processing.")
    parser.add_argument("--json_folder", type=str, help="Folder containing keypoints in .npy format.")

    args = parser.parse_args()
    process_data(args.data_folder, args.json_folder)

if __name__ == "__main__":
    main()
