from argparse import ArgumentParser
import cv2
from easymocap.mytools.reconstruction import batch_triangulate
# from easymocap.mytools.triangulator import batch_triangulate
from easymocap.mytools.triangulator import robust_triangulate_point, iterative_triangulate
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import scipy.io as sio
from glob import glob

# from tt import read_camera_from_mat



def undistort_pts(pts_2d, intrinsic, RD, TD):
    """
    Undistort 2D points based on camera intrinsic parameters and distortion coefficients.

    Parameters:
    pts_2d (numpy.ndarray): Array of 2D points in the image. Shape should be (N, 2).
    intrinsic (numpy.ndarray): Camera intrinsic matrix. Shape should be (3, 3).
    k1, k2, k3 (float): Radial distortion coefficients.
    p1, p2 (float): Tangential distortion coefficients.

    Returns:
    numpy.ndarray: Undistorted 2D points. Shape is (N, 2).
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    k1,k2 = RD
    p1,p2 = TD

    undistorted_pts = []

    for pt in pts_2d:
        print(pt)
        if np.isnan(pt).any():
            undistorted_pts.append([np.nan, np.nan])
            continue
        
        x = (pt[0] - cx) / fx
        y = (pt[1] - cy) / fy

        r2 = x**2 + y**2
        radial_distortion = 1 + k1 * r2 + k2 * r2**2

        x_corrected = x * radial_distortion + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_corrected = y * radial_distortion + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

        u_corrected = fx * x_corrected + cx
        v_corrected = fy * y_corrected + cy

        undistorted_pts.append([u_corrected, v_corrected])
        print(undistorted_pts[-1])

    return np.array(undistorted_pts)

def read_camera(mat_path, cam_num):
    param_mat = sio.loadmat(mat_path)['cameras']
    cameras = param_mat
        
    palls, ks, rts, RDs, TDs = [], [], [], [], []

    for cam_id in range(cam_num):
        param = param_mat[f"Camera{cam_id + 1}"][0,0]
        k = param['IntrinsicMatrix'][0,0].T
        r = param['rotationMatrix'][0,0].T
        t = param['translationVector'][0,0]
        RD = param['RadialDistortion'][0][0][0]
        TD = param['TangentialDistortion'][0][0][0]
        
        rt = np.hstack([r, t.T]).reshape(3, 4)
        palls.append(k@rt)
        
        ks.append(k)
        rts.append(rt)
        
        RDs.append(RD)
        TDs.append(TD)
        
    return cameras, np.array(palls), ks, rts, RDs, TDs

def read_cameras(mat_path, cam_num):

    mats = sorted(glob(f'{mat_path}/*'))
    print(mats)
        
    cameras, palls, ks, rts, RDs, TDs = [], [], [], [], [], []

    for cam_id in range(cam_num):
        cam = sio.loadmat(mats[cam_id])
        k = cam['K'].T
        r = cam['r'].T
        t = cam['t']
        RD = cam['RDistort']
        TD = cam['TDistort']
        
        rt = np.hstack([r, t.T]).reshape(3, 4)
        palls.append(k@rt)
        
        ks.append(k)
        rts.append(rt)
        
        RDs.append(RD)
        TDs.append(TD)
        cameras.append(cam)
        
    return cameras, np.array(palls), ks, rts, RDs, TDs

def load_com_2d(json_path):
    '''
    return: (n_cam, frame_num, joint_num, 3)
    '''
    
    # Load JSON data from the file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    kpt2d = []
    for key, item in data.items():
        print(np.array(item).shape)
        
        kpt2d.append(np.array(item))
    print(kpt2d)
    for i in kpt2d:
        print(i.shape)
    kpt2d = np.stack(kpt2d)
    print(kpt2d.shape)
    # add confidence
    new_values = np.ones((kpt2d.shape[0], kpt2d.shape[1], 1))
    kpt2d = np.concatenate((kpt2d, new_values), axis=2)
    kpt2d = np.expand_dims(kpt2d, axis=2)
    return kpt2d       
    
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument(
        '--cali_folder', '--cali_foler',
        dest='cali_folder',
        type=str,
        required=True,
        help='calibration folder path.')
    parser.add_argument(
        '--demo_folder', '--demo_foler',
        dest='demo_folder',
        type=str,
        required=True,
        help='input 2d kpts and output 3d kpts folder path.')
    
    args = parser.parse_args()
    
    cali_folder = args.cali_folder
    demo_folder = args.demo_folder

    kpt_num = 24
    cam_num = 8
    selected_cam = [0,1,2,3,4,5,6,7]
    cam_map = [0, 1, 2, 3, 4, 5, 6, 7]

    cameras, palls, ks, rts, RDs, TDs = read_cameras(cali_folder, cam_num)

    keypoints2d = []
    for cam in selected_cam:
        json_path = f'{demo_folder}/data{cam + 1}.json'
        f_kpts = []
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            for frame in tqdm(data):
                kpts = frame['instances'][0]['keypoints']
                scores = np.array(frame['instances'][0]['keypoint_scores']).reshape(-1, 1)
                kpts = np.concatenate((kpts, scores), axis=1)
                f_kpts.append(kpts)
        keypoints2d.append(f_kpts)
    keypoints2d = np.array(keypoints2d)

    # # Load camera parameters (you can modify this part based on your camera data)
    print("Load camera parameters")
    frame_num = keypoints2d.shape[1]
    keypoints3d = []

    dist_max = 200
    print("Triangulating...")
    for i in tqdm(range(frame_num)):
        if np.isnan(keypoints2d[selected_cam, i, :, :]).any():
            keypoints3d.append(np.full((24, 4), np.nan))
            continue
        
        frame, _  = iterative_triangulate(keypoints2d[selected_cam, i, :, :], palls, dist_max=dist_max, thres_outlier_view=1.0)
        
        if frame is None:
            frame = np.full((24, 4), np.nan)
        keypoints3d.append(frame)

    keypoints3d = np.array(keypoints3d)
    print(keypoints3d.shape)
    com = keypoints3d[:, :, :3]
    print(com.shape)
    print("Triangulation completed!")

    np.save(f'{demo_folder}/tri_new.npy', com)
