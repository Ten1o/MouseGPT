import os
import json
import numpy as np
from tqdm import tqdm

center = np.array((744.256112664111, 362.77046140691186))

radius = 239.96790996278034 / 2

json_path = os.environ['MOUSEGPT_KEYPOINTS_JSON']
with open(json_path, 'r') as f:
    data = json.load(f)

frame_num = len(data)

in_circle_num = 0
in_circle = []
for frame in tqdm(data):
    cantroid = np.array(frame['instances'][0]['keypoints'][9])

    distance = np.linalg.norm(cantroid - center)

    if distance < radius:
        in_circle.append(True)
        in_circle_num += 1
    else:
        in_circle.append(False)

print(f'incircle num: {in_circle_num}, total_num: {frame_num}, percentage:{in_circle_num/frame_num}')

in_circle = np.save(json_path.replace('json', 'npy').replace('.', '_tialend.'), np.array(in_circle), allow_pickle=False)

