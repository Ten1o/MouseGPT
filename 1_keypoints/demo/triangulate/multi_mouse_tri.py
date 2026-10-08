import os
import cv2
import json
import time
import pickle

from tqdm import tqdm
from glob import glob
from itertools import product
from collections import Counter

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from min_tri_mse_batch_tri import read_camera_from_mat, project

from easymocap.mytools.triangulator import robust_triangulate_point, iterative_triangulate, batch_triangulate

bones = [[ 1,  2],
        [ 2,  3],
        [ 1,  3],
        [ 1,  4],
        [ 2,  4],
        [ 3,  4],
        [ 4,  5],
        [ 7,  8],
        [ 8,  9],
        [ 9, 10],
        [11, 12],
        [12, 13],
        [13, 14],
        [14,  4],
        [15, 16],
        [16, 17],
        [17, 18],
        [18,  4],
        [19, 20],
        [20, 21],
        [22, 23],
        [23, 24],
        [ 6,  5],
        [ 6,  7],
        [ 7, 21],
        [ 7, 24]]

xmin, xmax, ymin, ymax, zmin, zmax = -0.02, 0.28, -0.02, 0.28, -0.02, 0.19

is_Zero = False
Zero_num = 0
Zero_list = []

def get_match_index(select_frames):
    select_cam = list(select_frames.keys())
    select_pose = np.array(list(select_frames.values()))

    # Full arrangement
    indices = list(product(*[range(len(lst)) for lst in select_pose if len(lst) > 0]))

    all_mse_list = []
    for idxs in indices[:len(indices) // 2]:

        # tri from 2d 8view
        cams = palls[select_cam]
        res  = batch_triangulate(select_pose[range(len(select_pose)), idxs, :, :], cams)

        # cal reprojection mse
        mse = 0
        for id, view_id in enumerate(select_cam):

            # cal reproject 2d point
            res_2d = project(res[:, :3], ks[view_id], rs[view_id], ts[view_id], RDs[view_id], TDs[view_id])
            res_2d = np.squeeze(res_2d, axis=1)

            # cal mse
            mse += np.sum((res_2d - select_frames[view_id][idxs[id], :, :2]) ** 2)

        all_mse_list.append([mse, np.array(idxs)])

    all_mse_list.sort(key=lambda x:x[0])

    mouse1 = batch_triangulate(select_pose[range(len(select_pose)), all_mse_list[0][1], :, :], cams)
    mouse2 = batch_triangulate(select_pose[range(len(select_pose)), 1-(all_mse_list[0][1]), :, :], cams)

    return [mouse1, mouse2]


def draw_3d_kpts(ax, points, name):

    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')

    for bone in bones:
        x = [points[bone[0]-1][0], points[bone[1]-1][0]]
        y = [points[bone[0]-1][1], points[bone[1]-1][1]]
        z = [points[bone[0]-1][2], points[bone[1]-1][2]]
        ax.plot(x, y, z, c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.view_init(elev=10, azim=50)


def update(frame_id):
    # print(f'frame {frame_id}', end='\r')

    global is_Zero, Zero_num, Zero_list

    select_frames = {}
    for cam_id in range(cam_num):
        if len(data_2D[cam_id][frame_id]) == 2:
            select_frames[cam_id] = np.array(data_2D[cam_id][frame_id])

    if len(select_frames) < 3:
        if is_Zero:
            Zero_num += 1
        else:
            is_Zero = True
            Zero_num = 1
        kpts_3d = np.zeros((2,24,4))
    else:
        if is_Zero:
            Zero_list.append(Zero_num)
            is_Zero = False
        kpts_3d = get_match_index(select_frames)

    return kpts_3d

    # ax.clear()

    # for id, mouse in enumerate(kpts_3d):
    #     draw_3d_kpts(ax, mouse, f'{frame_id}-{id}')


if __name__ == '__main__':
    data_folder = os.environ['MOUSEGPT_SESSION_DIR']

    # jsons = []
    # for json_name in sorted(glob(f'{data_folder}/*.json')):
    #     print(json_name)

    #     with open(json_name, 'r') as f:
    #         raw_data = json.load(f)
    #         jsons.append(raw_data)

    # # Load 2D keypoints
    # data_2D = []

    # for raw_data in jsons:

    #     data = []
    #     for raw_frame in tqdm(raw_data):

    #         frame = []
    #         for instance in raw_frame['instances']:
    #             kpts = np.array(instance['keypoints'])
    #             score = np.array(instance['keypoint_scores'])
    #             kpts_with_score = np.concatenate((kpts, score.reshape(len(score), -1)), axis=1)

    #             frame.append(kpts_with_score)

    #         data.append(frame)

    #     data_2D.append(data)

    with open("data_2d_free_2.pkl", "rb") as f:
        data_2D = pickle.load(f)

    # read camera
    mat_path = os.environ['MOUSEGPT_CALIBRATION_MAT']
    cameras, palls, ks, rts, RDs, TDs, rs, ts = read_camera_from_mat(mat_path)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    frame_num = len(data_2D[0])
    cam_num = len(data_2D)

    data_3D = []
    for frame_id in tqdm(range(frame_num)):
        kpts_3d = update(frame_id)
        data_3D.append(kpts_3d)

    data_3D = np.array(data_3D)[..., :3]

    np.save('multi_mouse_free_2.npy', data_3D)

    plt.plot(range(len(Zero_list)), Zero_list)

    plt.savefig('zero_free_2.png')

    print(max(Zero_list))

    # ani = FuncAnimation(fig, update, frames=range(frame_num), interval=1000/60)


    # ani.save('animation-all-2.mp4', writer='ffmpeg')


    # plt.show()
