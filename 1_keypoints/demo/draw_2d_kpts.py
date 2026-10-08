import os
import cv2
import json
import numpy as np


with open(os.environ['MOUSEGPT_KEYPOINTS_JSON'], "r") as f:
    keypoints_data = json.load(f)


colors = [[1., 0., 0., 1],
          [1, 0.26086957, 0, 1],
          [1, 0.52173913, 0, 1],
          [1, 0.7826087, 0, 1],
          [0.95652174, 1, 0, 1],
          [0.69565217, 1, 0, 1],
          [0.43478261, 1, 0, 1],
          [0.17391304, 1, 0, 1],
          [0, 1, 0.08695652, 1],
          [0, 1, 0.34782609, 1],
          [0, 1, 0.60869565, 1],
          [0, 1, 0.86956522, 1],
          [0, 0.86956522, 1, 1],
          [0, 0.60869565, 1, 1],
          [0, 0.34782609, 1, 1],
          [0, 0.08695652, 1, 1],
          [0.17391304, 0, 1, 1],
          [0.43478261, 0, 1, 1],
          [0.69565217, 0, 1, 1],
          [0.95652174, 0, 1, 1],
          [1, 0, 0.52173913, 1],
          [1, 0, 0.26086957, 1],
          [1, 0, 1, 1],
          [1, 0, 1, 1],
          [1, 0, 0, 1],
          [1, 0, 0, 1]]


bones = [[1, 2], [2, 3], [1, 3], [1, 4], [2, 4], [3, 4], [4, 5], [7, 8], [8, 9],
         [9, 10], [11, 12], [12, 13], [13, 14], [14, 4], [15, 16], [16, 17],
         [17, 18], [18, 4], [19, 20], [20, 21], [22, 23], [23, 24], [6, 5],
         [6, 7], [7, 21], [7, 24]]


cap = cv2.VideoCapture(os.environ['MOUSEGPT_VIDEO'])


frame_to_draw = 756
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_draw)


ret, frame = cap.read()
if ret:

    frame_data = keypoints_data[frame_to_draw]
    instances = frame_data["instances"]

    for instance in instances:
        keypoints = instance["keypoints"]
        keypoints = np.array(keypoints, dtype=np.int32)


        for x, y in keypoints:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


        for idx, (start, end) in enumerate(bones):
            color = (int(colors[idx][0] * 255), int(colors[idx][1] * 255), int(colors[idx][2] * 255))
            start_point = tuple(keypoints[start - 1])
            end_point = tuple(keypoints[end - 1])
            cv2.line(frame, start_point, end_point, color, 2)


    output_path = '240517_C13M3_data5_756.png'
    cv2.imwrite(output_path, frame)


cap.release()
