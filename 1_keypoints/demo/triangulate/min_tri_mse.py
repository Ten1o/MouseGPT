import os
import cv2
import json
import time

from glob import glob
from itertools import product

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from easymocap.mytools.triangulator import robust_triangulate_point, iterative_triangulate


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

def read_camera_from_mat(mat_path):
    param_mat = sio.loadmat(mat_path)['camParams']
    cameras = param_mat

    palls, ks, rts, rs, ts = [], [], [], [], []
    RDs, TDs = [], []
    for cam_id in range(param_mat.shape[0]):
        k = param_mat[cam_id, 0]['K'][0,0].T
        r = param_mat[cam_id, 0]['r'][0,0].T
        t = param_mat[cam_id, 0]['t'][0,0]
        RD = param_mat[cam_id, 0]['RDistort'][0][0][0]
        TD = param_mat[cam_id, 0]['TDistort'][0][0][0]

        rt = np.hstack([r, t.T]).reshape(3, 4)
        palls.append(k@rt)

        ks.append(k)
        rts.append(rt)
        rs.append(r)
        ts.append(t)

        RDs.append(RD)
        TDs.append(TD)

    return cameras, np.array(palls), np.array(ks), np.array(rts), np.array(RDs), np.array(TDs), np.array(rs), np.array(ts)

def grab_one_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    return cap

def update(ax, points, mse):

    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')

    for bone in bones:
        x = [points[bone[0]-1][0], points[bone[1]-1][0]]
        y = [points[bone[0]-1][1], points[bone[1]-1][1]]
        z = [points[bone[0]-1][2], points[bone[1]-1][2]]
        ax.plot(x, y, z, c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=10, azim=50)

def project(p3d, k, r, t, RD, TD):
    p3d = p3d.reshape(-1, 1, 3)
    points_2d, _ = cv2.projectPoints(p3d, r, t, k, np.hstack([RD, TD]))
    p2d = tuple(map(int, points_2d[0][0]))
    return p2d

mats = sorted(glob(os.environ['MOUSEGPT_CALIBRATION_GLOB']))

for mat_path in mats:

    B_mat_path = mat_path.replace('A', 'B')
    mat_name = mat_path.split('/')[-1]

    if 'A' not in mat_name or not os.path.exists(B_mat_path):
        continue

    date = mat_name[:6]
    group_name = mat_name.split('_')[1]
    v2 = '_2' if 'v2' in mat_name else ''

    root_folder = mat_name.split('.')[0]
    if root_folder[-1] != '0':
        root_folder = root_folder[:-1]

    data_folder = os.environ['MOUSEGPT_DATA_ROOT'] + f'/20{date}/{date}_{group_name}/demo{v2}/{root_folder}'

    # read camera
    cameras, palls, ks, rts, RDs, TDs, rs, ts = read_camera_from_mat(mat_path)

    # read 3D keypoints
    data_3D = []
    data_3D.append(sio.loadmat(mat_path)['data_3D'])
    if 'A' in mat_path:
        data_3D.append(sio.loadmat(mat_path.replace('A', 'B'))['data_3D'])

    # read 2D keypoints
    data_2D = []
    data_2D.append(sio.loadmat(mat_path)['data_2D']) # (24, 8, 2, 10)
    if 'A' in mat_path:
        data_2D.append(sio.loadmat(mat_path.replace('A', 'B'))['data_2D']) # (24, 8, 2, 10)

    dist_max = 200

    frame_num = data_2D[0].shape[3]

    data = np.array(data_2D).transpose(4,2,0,1,3)

    score = np.ones((10,8,2,24,1))
    # 10x8x2x24x3
    data = np.concatenate((data, score), axis=4)

    mse = 0
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # update(ax, data_3D[0][0], mse)
    # update(ax, data_3D[1][0], mse)

    # plt.savefig(f'{mse:.3f}.jpg')



    for i in range(frame_num):
        start = time.time()
        print('start', start)

        # Full arrangement
        indices = list(product(*[range(len(lst)) for lst in data[i] if len(lst) > 0]))

        # min_match = [[], []]
        # min_mse = np.ones(2)
        all_mse_list = []
        for idxs in indices[:len(indices) // 2]:

            # tri from 2d 8view
            res, _  = iterative_triangulate(data[i, range(data.shape[1]), idxs, :, :], palls, dist_max=dist_max, thres_outlier_view=1.0)

            # cal 3d gt mse
            # mse = np.zeros(2)
            # for j in range(2):
            #     mse[j] = np.sum((res[:, :3] - data_3D[j][0]) ** 2)
            # match_id = np.argmin(mse)
            # mse = mse[match_id]

            # cal reprojection mse
            mse = 0
            for view_id in range(8):
                # cal reproject 2d point
                res_2d = project(res, ks[view_id], rs[view_id], ts[view_id], RDs[view_id], TDs[view_id])

                # cal mse
                mse += np.sum((res_2d - data[i, view_id, idxs[view_id], :, :2]) ** 2)

            all_mse_list.append([mse, idxs])

            # if mse < min_mse[match_id]:
            #     min_mse[match_id] = mse
            #     min_match[match_id] = idxs

            # ax.clear()
            # update(ax, res, mse)
            # plt.savefig(f'{idxs}-{mse:.8f}.jpg')

        all_mse_list.sort(key=lambda x:x[0])

        print(all_mse_list[0])

        end = time.time()
        print('end', end)
        print('delta', end - start)

    quit()
