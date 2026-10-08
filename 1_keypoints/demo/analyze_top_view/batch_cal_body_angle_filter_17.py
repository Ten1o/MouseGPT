import os
import json
import numpy as np
from tqdm import tqdm
import cv2
from glob import glob
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt


def lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y




def filter(kpts):
    # Apply the Savitzky-Golay filter to the 3D keypoint data
    window_length = 40
    poly_order = 7

    # Ensure window length is odd
    if window_length % 2 == 0:
        window_length += 1

    filtered_data_3d = np.empty_like(kpts)

    # Apply the filter for each dimension separately
    for point in range(kpts.shape[1]):
        for dim in range(2):
            filtered_data_3d[:, point, dim] = savgol_filter(kpts[:, point, dim], window_length, poly_order)
    return filtered_data_3d

data_root = os.environ['MOUSEGPT_DATA_ROOT']
out_root = './body_angle'

for day in [value.strip() for value in os.environ['MOUSEGPT_DATES'].split(',') if value.strip()]:
    for group in sorted(glob(f"{data_root}/{day}/{os.environ['MOUSEGPT_GROUP_GLOB']}")):
        print(group)
        print('Loading keypoints json ……')

        json_path = f'{group}/data1.json'
        with open(json_path, 'r') as f:
            data = json.load(f)

        frame_num = len(data)
        print(f'total {frame_num} frames')

        kpts = []

        for frame in tqdm(data, desc='loading kpts'):
            kpt = np.array(frame['instances'][0]['keypoints'])
            kpts.append(kpt)

        kpts = np.array(kpts)
        print(kpts.shape)

        filtered_kpts = filter(kpts)



        head_direction = filtered_kpts[:,2,:] - filtered_kpts[:,3,:]
        body_direction = filtered_kpts[:,3,:] - filtered_kpts[:,4,:]


        dot_products = np.einsum('ij,ij->i', head_direction, body_direction)


        norms1 = np.linalg.norm(head_direction, axis=1)
        norms2 = np.linalg.norm(body_direction, axis=1)


        cos_theta = dot_products / (norms1 * norms2)


        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.arccos(cos_theta)


        cross_products = body_direction[:, 0] * head_direction[:, 1] - body_direction[:, 1] * head_direction[:, 0]


        angles = np.where(cross_products < 0, -angles, angles)


        angle_deg = np.degrees(angles)
        np.save(f'{out_root}/{group.split("/")[-1]}.npy', angle_deg)


        filtered_deg = lowpass_filter(angle_deg, cutoff=1, fs=120, order=3)


        abs_angle_deg = np.abs(angle_deg)
        bin_size = len(abs_angle_deg) // 100
        sums = np.array([np.sum(abs_angle_deg[i*bin_size:(i+1)*bin_size]) for i in range(100)])


        plt.figure(figsize=(10, 5))
        plt.plot(sums, marker='o', linestyle='-')
        plt.title('Sum of degree')
        plt.xlabel('Time(60min/100)')
        plt.ylabel('Sum of Elements')
        plt.grid(True)
        plt.savefig(f'{out_root}/{group.split("/")[-1]}_100.png')

        abs_angle_deg = np.abs(angle_deg)
        bin_num = len(abs_angle_deg) // 3600
        bin_size = 3600
        sums = np.array([np.sum(abs_angle_deg[i*bin_size:(i+1)*bin_size]) for i in range(bin_num)])


        plt.figure(figsize=(10, 5))
        plt.plot(sums, marker='o', linestyle='-')
        plt.title('Sum of degree')
        plt.xlabel('Time(min)')
        plt.ylabel('Sum of Elements')
        plt.grid(True)
        plt.savefig(f'{out_root}/{group.split("/")[-1]}_min.png')
