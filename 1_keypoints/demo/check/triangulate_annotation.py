import scipy.io
import numpy as np
from glob import glob
import cv2
import argparse
import pandas as pd
import os
from easymocap.mytools.triangulator import iterative_triangulate



def read_camera_from_mat(mat_path):
    param_mat = scipy.io.loadmat(mat_path)['camParams']
    cameras = param_mat
        
    palls, ks, rts = [], [], []
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
  
        RDs.append(RD)
        TDs.append(TD)
  
    return cameras, np.array(palls), ks, rts, RDs, TDs


def project(p3d, k, r, t, RD, TD):
    p3d = p3d.reshape(-1, 1, 3)
    # print(RD, TD)
    RD[2] = 0
    TD *= 0
    # print(RD, TD)
    points_2d, _ = cv2.projectPoints(p3d, r, t, k, np.hstack([RD, TD]))
    p2d = tuple(map(int, points_2d[0][0]))
    return p2d

def convert_data2d(data_2D):
    '''
    convert data2d for triangulation
    data_2D: (n_kpt, n_view, 2, n_frame)
    new_data_2D: (n_view, n_frame, n_kpt, 3) with confidence
    '''
    
    data_2D = data_2D.transpose(1, 3, 0, 2)
    new_data_2D = np.ones((data_2D.shape[0], data_2D.shape[1], data_2D.shape[2], 3))
    new_data_2D[..., :2] = data_2D
    return new_data_2D


def main(args):
    dist_max = 200
    
    label_paths = sorted(glob(f'{args.label_folder}/*'))
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)
    
    for label_path in label_paths:
        filename = os.path.basename(label_path)
        start_frame = int(filename[-8:-4])
        print(filename)
        
        # read camera
        cameras, palls, ks, rts, RDs, TDs = read_camera_from_mat(label_path)
        
        # read keypoints
        data_2D = scipy.io.loadmat(label_path)['data_2D'] # (24, 8, 2, 10)
        new_data_2D = convert_data2d(data_2D)
        data_2D_backup = scipy.io.loadmat(label_path)['data_2D_Backup'] # (24, 8, 2, 10)
        
        kpt_num = data_2D.shape[0]
        frame_num = data_2D.shape[-1]
        cam_num = len(cameras)

        data_3D = scipy.io.loadmat(label_path)['data_3D'].reshape(-1, kpt_num, 3) # (10, 72)
        new_data_3D = np.zeros_like(data_3D) * np.nan
        for frame_id in range(frame_num):
            # for kpt_idx in range(data_3D.shape[1]):
                # x = data_3D[frame_id, kpt_idx]
                # if np.isnan(x).any():
                    # print(f"3D frame num={frame_id}, kpt_idx={kpt_idx}", point)
                    # keypoints2d[selected_cam, i, :, :]
            try:
                triangulate_3d, _  = iterative_triangulate(new_data_2D[:, frame_id, :, ], palls, dist_max=dist_max, thres_outlier_view=1.0)
                new_data_3D[frame_id] = triangulate_3d[:, :3]
            except np.linalg.LinAlgError:
                pass 
            
            for view_cam in range(cam_num):        
                for kpt_idx in range(data_2D.shape[0]):
                    point = data_2D[kpt_idx, view_cam, :, frame_id]
                    if np.isnan(point).any():
                        print(f"2D frame num={frame_id}, cam_num={view_cam}, kpt_idx={kpt_idx}", kpt_idx, point)

        new_data_3D = new_data_3D.reshape(frame_num, -1, 3)
        new_mat = scipy.io.loadmat(label_path)
        new_mat['data_3D'] = new_data_3D
        save_mat_path = f'{save_folder}/{filename}'
        scipy.io.savemat(save_mat_path, new_mat)
        
        
if __name__ == '__main__':            
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_folder", type=str, required=True, help="Path to .mat folder")
    parser.add_argument("--save_folder", type=str, required=True, help="Path to new .mat folder")
    
    args = parser.parse_args()
    main(args)
    