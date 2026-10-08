import json
import cv2
import numpy as np
from tqdm import tqdm
from tt import write_json



        
demo_folder = '/home/vrlab/projects/mouse/dannce/demo/mouse230919_30s'
json_path = f'{demo_folder}/COM/yolo_results/yolo_com.json'
visualize = False
cam_num = 8
com_data_all = {}

for i in range(0, cam_num):
    cam_id = i + 1

    # Load JSON data from the file
    with open(f'{demo_folder}/bbox/Camera{cam_id}_bbox.json', 'r') as json_file:
        data = json.load(json_file)
        
    # Input video file
    input_video_path = f'{demo_folder}/videos/Camera{cam_id}/0.mp4'
    downscale = 8
    cap = cv2.VideoCapture(input_video_path)

    # Initialize tqdm progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if visualize:
        # Output video file with markers
        output_video_path = f'{demo_folder}/Camera{cam_id}_com.mp4'

        # Open the input video

        # Get video properties (frame width, height, and frames per second)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        

    # Loop through each frame in the input video
    com_data = []
    for frame_id in tqdm(range(total_frames), desc=f"Camera{cam_id}"):
        if visualize:
            ret, frame = cap.read()
            if not ret:
                break

        # Extract bounding box information for the current frame
        frame_data = data.get(str(frame_id+1))
        if frame_data:
            bbox_data = frame_data['pred_instances']['bboxes']

            com = (0.0, 0.0)
            
            # Loop through each bounding box in the frame
            for bbox in bbox_data:
                x_min, y_min, x_max, y_max = map(int, bbox)
                center_x = ((x_min + x_max) // 2) * downscale
                center_y = ((y_min + y_max) // 2) * downscale
                com = (center_x, center_y)
            
                
                if visualize:
                    # Add a marker at the center of the bounding box
                    cv2.circle(frame, com, 5, (0, 0, 255), -1)
                break # just draw one bbox with the highest confidence

            com_data.append(com)
        # Write the frame with markers to the output video
        if visualize:
            out.write(frame)

    com_data_all[f'Camera{cam_id}'] = com_data
    write_json(json_path, com_data_all)
    # Release video objects
    if visualize:
        cap.release()
        out.release()
        print(f"Video with markers saved as {output_video_path}")
