import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob


def merge_frame_write(frame_list, out_vid):
    h = [cv2.hconcat(frame_list[i*4:i*4+4]) for i in range(2)]
    v = cv2.vconcat(h).resize((1920, 1080))

    out_vid.write(v)


data_2D = np.load('multi_mouse_2.npy')

indices = np.where((data_2D==0).all(axis=(1,2,3)))[0]


is_Zero = False
Zero_num = 0
Zero_list = []
first_zero = 0

print(len(indices))

for index in range(1, len(indices)):
    if indices[index] - 1 == indices[index - 1]:
        if is_Zero:
            Zero_num += 1
        else:
            is_Zero = True
            first_zero = indices[index]
            Zero_num = 1
    else:
        if is_Zero:
            if Zero_num>100:
                print(Zero_num, first_zero)
            Zero_list.append(Zero_num)
            is_Zero = False

print(len(Zero_list))
fig = plt.figure()

plt.plot(range(len(Zero_list)), Zero_list)

plt.savefig('zero_2.png')

folder_path = os.environ['MOUSEGPT_SESSION_DIR']

# caps = [cv2.VideoCapture(vid) for vid in sorted(glob(f'{folder_path}/data*'))]
# print(caps)

# is_Zero = False
# Zero_num = 0
# Zero_list = []


# fourc = cv2.VideoWriter_fourcc(*'XVID')
# fps = 30
# height = 1080
# width = 1920
# out_vid = None

# for index in tqdm(range(len(indices)-1)):
#     if indices[index] + 1 == indices[index + 1]:
#         if is_Zero:
#             frame_list = []
#             for cap in caps:
#                 _, frame = cap.read()
#                 frame_list.append(frame)
#             merge_frame_write(frame_list, out_vid)
#         else:
#             out_vid = cv2.VideoWriter(f'{out_folder}/{index}.mp4', fourc, fps, (width, height))
#             frame_list = []
#             for cap in caps:
#                 cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#                 _, frame = cap.read()
#                 print(frame.shape)
#                 frame_list.append(frame)
#             merge_frame_write(frame_list, out_vid)
#     else:
#         if is_Zero:
#             out_vid.release()
#             is_Zero = False
