import os
import json
import numpy as np
from tqdm import tqdm
from glob import glob

center = np.array((0.12, 0.13))

radius = 0.12 / 2

day = os.environ['MOUSEGPT_DATE']
root_folder = os.environ['MOUSEGPT_DATA_ROOT']

for group in sorted(glob(f"{root_folder}/{day}/{os.environ['MOUSEGPT_GROUP_GLOB']}")):
    npy_path = f'{group}/tri_filter.npy'
    data = np.load(npy_path)

    frame_num = len(data)

    in_circle_num = 0
    in_circle = []
    for frame in data:
        cantroid = frame[4][:2]

        distance = np.linalg.norm(cantroid - center)

        if distance < radius:
            in_circle.append(True)
            in_circle_num += 1
        else:
            in_circle.append(False)

    print(f'{group.split("/")[-1]}, incircle num: {in_circle_num}, total_num: {frame_num}, percentage:{in_circle_num/frame_num}')

    in_circle = np.save(npy_path.replace('tri_filter', 'center'), np.array(in_circle), allow_pickle=False)

