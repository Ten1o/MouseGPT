import cv2
from easymocap.mytools.reconstruction import batch_triangulate
# from easymocap.mytools.triangulator import batch_triangulate
from easymocap.mytools.triangulator import robust_triangulate_point

import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import scipy.io as sio

from tt import read_camera_from_mat


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
 
def main():
    demo_folder = f'/home/vrlab/projects/mouse/dannce/demo/mouse230919_30s_v2'
    kpt_num = 1
    cam_num = 8
    selected_cam = [0,1,2,3,4,5,6,7]

    # print("Load 2D center of mass points")
    keypoints2d= load_com_2d(f'{demo_folder}/COM/yolo_com/yolo_com.json')
    # # Load camera parameters (you can modify this part based on your camera data)
    print("Load camera parameters")
    cameras, palls, ks, rts = read_camera_from_mat(f'{demo_folder}/20230927_191310_Label3D_dannce.mat')
    frame_num = keypoints2d.shape[1]

    keypoints3d = []
    print("Triangulating...")
    for i in range(frame_num):
        # Perform triangulation for the current frame
        # keypoints3d_frame = batch_triangulate(keypoints2d[selected_cam, i, :, :], palls[selected_cam])
        index, keypoints3d_frame = robust_triangulate_point(keypoints2d[selected_cam, i, :, :], palls[selected_cam], dist_max=100)
        print(i, keypoints3d_frame.shape)
        keypoints3d_frame = keypoints3d_frame.reshape(1,-1)
        keypoints3d.append(keypoints3d_frame)
    com = np.array(keypoints3d)[:, 0, :3]
    print("Triangulation completed!")
    
    

    # Sample data
    sampleID = np.arange(1, frame_num + 1, dtype=np.int64)

    # Create a dictionary to hold the data
    data_dict = {
        'sampleID': sampleID,
        'com': com,
    }

    # Save the data to a .mat file
    sio.savemat(f'{demo_folder}/COM/yolo_results/com3d.mat', data_dict)

    
    # Further processing or visualization can be added here based on your needs
    

if __name__ == "__main__":
    main()
