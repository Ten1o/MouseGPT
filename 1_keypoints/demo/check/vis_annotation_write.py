import scipy.io
import numpy as np
from glob import glob
import cv2
import argparse
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
    # TD *= 0
    # print(RD, TD)
    points_2d, _ = cv2.projectPoints(p3d, r, t, k, np.hstack([RD, TD]))
    p2d = tuple(map(int, points_2d[0][0]))
    return p2d


def grab_one_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    return cap


def draw_skeleton(background_img, pts, skeleton, alpha):
    color = skeleton['color'][0][0] # (26, 4) (num connections, rgba)
    connections = skeleton['joints_idx'][0][0] # (26, 2) (num connections, joints indices)
    for c, conn in zip(color, connections):
        joint_idx1, joint_idx2 = conn

        if np.isnan(pts[joint_idx1-1]).any() or np.isnan(pts[joint_idx2-1]).any():
            continue
        point1 = tuple(map(int, pts[joint_idx1-1]))
        point2 = tuple(map(int, pts[joint_idx2-1]))

        c = np.array([c[2], c[1], c[0], c[3]]) * 255
        # Draw the line with the specified color and transparency
        thickness = 10
        overlay = background_img.copy()
        cv2.line(background_img, point1, point2, c, thickness)  # Adjust line thickness as needed
        background_img = cv2.addWeighted(overlay, alpha, background_img, 1-alpha, 0)
        
    return background_img
    
    
def draw_point(uv, text, background_img, pt_color=(255, 255, 0), transparency=0.5, draw_text=True):
    if uv[0] < 0 or uv[0] >= background_img.shape[1] or uv[1] < 0 or uv[1] >= background_img.shape[0]:
        return background_img
    uv = tuple(map(int, uv))
    point_size = 10
    overlay = background_img.copy()
    cv2.circle(background_img, uv, point_size, pt_color, -1)
    background_img = cv2.addWeighted(overlay, transparency, background_img, 1 - transparency, 0)

    if draw_text:
        # Define the font and text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        line_thickness = 3

        # Put the text on the image
        background_img = cv2.putText(background_img, text, uv, font, font_scale, font_color, line_thickness)

    return background_img



def main(args):
    ofolder = args.ofolder
    os.makedirs(ofolder, exist_ok=True)
    
    label_path = args.label_path
    
    # read videos
    video_paths = sorted(glob(f'{args.data_folder}/videos/*/*.mp4'))
    caps = [grab_one_frame(vid) for vid in video_paths]

    # read camera
    cameras, palls, ks, rts, RDs, TDs = read_camera_from_mat(args.label_path)
    
    # read skeleton
    skeleton = scipy.io.loadmat(label_path)['skeleton']
    print(skeleton['color'][0][0])
    print('-------------------')
    print(skeleton['joints_idx'][0][0])
    
    # read keypoints
    data_2D = scipy.io.loadmat(label_path)['data_2D'] # (24, 8, 2, 10)
    data_2D_backup = scipy.io.loadmat(label_path)['data_2D_Backup'] # (24, 8, 2, 10)
    
    kpt_num = data_2D.shape[0]
    frame_num = data_2D.shape[-1]
    cam_num = len(cameras)

    data_3D = scipy.io.loadmat(label_path)['data_3D'].reshape(-1, kpt_num, 3) # (10, 72)

        
    # loop begin
    frame_id = 0
    view_cam = 0
    
    show_2d = True
    show_3d = False
    draw_text = False

    for frame_id in range(frame_num):
        for view_cam in range(cam_num):
            # print(f"frame_id={frame_id}, view_cam={view_cam}")
            cap = caps[view_cam]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, background_image = cap.read()

            k = np.array(ks[view_cam])
            r = rts[view_cam][:, :3]   # Extracting rotation matrix and transposing
            t = rts[view_cam][:, 3:4]   # Extracting translation matrix
            RD = np.array(RDs[view_cam])
            TD = np.array(TDs[view_cam])
            
            if show_2d:
                # draw 2D skeleton
                background_image = draw_skeleton(background_image, data_2D[:, view_cam, :, frame_id], skeleton, alpha=0.7)
                
                # draw 2D points
                for kpt_idx in range(data_2D.shape[0]):
                    point = data_2D[kpt_idx, view_cam, :, frame_id]
                    if np.any(np.isnan(point)):
                        print(f"2D frame num={frame_id}, cam_num={view_cam}, kpt_idx={kpt_idx}", kpt_idx, point)
                        continue
                    background_image = draw_point(point, f'{kpt_idx+1}', background_image, pt_color=(255,255,255), transparency=0.6, draw_text=draw_text)    
            
            if show_3d:
                for kpt_idx in range(data_3D.shape[1]):
                    x = data_3D[frame_id, kpt_idx]
                    if np.any(np.isnan(x)):
                        print(f"3D frame num={frame_id}, cam_num={view_cam}, kpt_idx={kpt_idx}", point)
                        continue
                    point = project(x, k, r, t, RD, TD)
                    background_image = draw_point(point, f'{kpt_idx+1}', background_image, pt_color=(255,255,0), transparency=0.6, draw_text=draw_text)
                    
            h, w, c = background_image.shape
            show_image = cv2.resize(background_image, (w//2, h//2))
            caption = f"frame: {frame_id+1}"
            cv2.putText(show_image, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            caption = f"cam: {view_cam+1}"
            cv2.putText(show_image, caption, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            caption = f"show_2d: {show_2d}"
            cv2.putText(show_image, caption, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            caption = f"show_3d: {show_3d}"
            cv2.putText(show_image, caption, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            caption = f"draw_text: {draw_text}"
            cv2.putText(show_image, caption, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            folder_name = os.path.basename(label_path).split('.')[0]
            caption = f"label_path: {folder_name}"
            cv2.putText(show_image, caption, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(f"{ofolder}/{folder_name}_frame{frame_id+1:0>2d}_cam{view_cam+1}.jpg", show_image)
            
            
if __name__ == '__main__':            
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", type=str, required=True, help="Path to .mat file")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to source data folder")
    parser.add_argument("--ofolder", type=str, required=True, help="Path to output data folder")
    
    args = parser.parse_args()
    main(args)
    