import scipy.io
import numpy as np
from glob import glob
import cv2
import argparse
import pandas as pd
import os

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


def main(args):
    
    columns_2d = ['folder_name', 'frame_id', 'kpt_id', 'cam_id']
    df_2d_data = []

    columns_3d = ['folder_name', 'frame_id', 'kpt_id']
    df_3d_data = []
    label_paths = sorted(glob(f'{args.label_folder}/*'))
    for label_path in label_paths:
        filename = os.path.basename(label_path)
        start_frame = int(filename[-8:-4])
        
        # read camera
        cameras, palls, ks, rts, RDs, TDs = read_camera_from_mat(label_path)
        
        # read keypoints
        data_2D = scipy.io.loadmat(label_path)['data_2D'] # (24, 8, 2, 10)
        data_2D_backup = scipy.io.loadmat(label_path)['data_2D_Backup'] # (24, 8, 2, 10)
        
        kpt_num = data_2D.shape[0]
        frame_num = data_2D.shape[-1]
        cam_num = len(cameras)

        data_3D = scipy.io.loadmat(label_path)['data_3D'].reshape(-1, kpt_num, 3) # (10, 72)

        # Initialize an empty DataFrame with defined column names
        for frame_id in range(frame_num):
            for kpt_idx in range(data_3D.shape[1]):
                x = data_3D[frame_id, kpt_idx]
                if np.isnan(x).any():
                    # print(f"3D frame num={frame_id}, kpt_idx={kpt_idx}", x)
                    frame_data = {"folder_name":filename, "frame_id": frame_id+1, "kpt_id": kpt_idx+1}
                    df_3d_data.append(frame_data)
                    # df_3d = df_3d.append(frame_data, ignore_index=True)
            
            for view_cam in range(cam_num):        
                for kpt_idx in range(data_2D.shape[0]):
                    point = data_2D[kpt_idx, view_cam, :, frame_id]
                    if np.isnan(point).any():
                        # print(f"2D frame num={frame_id}, cam_num={view_cam}, kpt_idx={kpt_idx}", kpt_idx, point)
                        frame_data = {"folder_name":filename, "frame_id": frame_id+1,  "kpt_id": kpt_idx+1, "cam_id":view_cam+1}
                        df_2d_data.append(frame_data)
                        # df_2d = df_2d.append(frame_data, ignore_index=True)
            
    df_3d = pd.DataFrame(df_3d_data, columns=columns_3d)
    df_2d = pd.DataFrame(df_2d_data, columns=columns_2d)

    excel_filename = "keypoint_stats_2d.xlsx"
    df_2d.to_excel(excel_filename, index=False)
    print(f"Data saved to {excel_filename}")

            
    excel_filename = "keypoint_stats_3d.xlsx"
    df_3d.to_excel(excel_filename, index=False)
    print(f"Data saved to {excel_filename}")

if __name__ == '__main__':            
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_folder", type=str, required=True, help="Path to .mat folder")
    
    args = parser.parse_args()
    main(args)
    