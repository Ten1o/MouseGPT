import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from scipy.signal import savgol_filter

colors = [[1., 0., 0., 1],
          [1., 0.26086957, 0., 1],
          [1., 0.52173913, 0., 1],
          [1., 0.7826087, 0., 1],
          [0.95652174, 1., 0., 1],
          [0.69565217, 1., 0., 1],
          [0.43478261, 1., 0., 1],
          [0.17391304, 1., 0., 1],
          [0., 1., 0.08695652, 1],
          [0., 1., 0.34782609, 1],
          [0., 1., 0.60869565, 1],
          [0., 1., 0.86956522, 1],
          [0., 0.86956522, 1., 1],
          [0., 0.60869565, 1., 1],
          [0., 0.34782609, 1., 1],
          [0., 0.08695652, 1., 1],
          [0.17391304, 0., 1., 1],
          [0.43478261, 0., 1., 1],
          [0.69565217, 0., 1., 1],
          [0.95652174, 0., 1., 1],
          [1., 0., 0.52173913, 1],
          [1., 0., 0.26086957, 1],
          [1., 0., 1., 1],
          [1., 0., 1., 1],
          [1., 0., 0., 1],
          [1., 0., 0., 1]]

bones = [[1, 2],
         [2, 3],
         [1, 3],
         [1, 4],
         [2, 4],
         [3, 4],
         [4, 5],
         [7, 8],
         [8, 9],
         [9, 10],
         [11, 12],
         [12, 13],
         [13, 14],
         [14, 4],
         [15, 16],
         [16, 17],
         [17, 18],
         [18, 4],
         [19, 20],
         [20, 21],
         [22, 23],
         [23, 24],
         [6, 5],
         [6, 7],
         [7, 21],
         [7, 24]]

def filter(data_3d):
    window_length = 15
    poly_order = 5

    if window_length % 2 == 0:
        window_length += 1

    filtered_data_3d = np.empty_like(data_3d)
    for dim in range(3):
        for point in range(data_3d.shape[1]):
            filtered_data_3d[:, point, dim] = savgol_filter(data_3d[:, point, dim], window_length, poly_order)
    return filtered_data_3d


npy_path = os.environ['MOUSEGPT_KEYPOINTS_NPY']
data_3d = np.load(npy_path)

frame_num = data_3d.shape[0]
if data_3d.shape[-1] == 3:
    data_3d = data_3d.reshape(frame_num, 24, 3)
else:
    data_3d = data_3d[:, :, :, :3].reshape(frame_num, 24, 3)


data_3d = filter(data_3d)


frame_to_plot = 756
points = data_3d[frame_to_plot]

xmin, xmax, ymin, ymax, zmin, zmax = -0.02, 0.28, -0.02, 0.28, -0.02, 0.19

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')


for bone, color in zip(bones, colors):
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
ax.set_title(f'Frame {frame_to_plot}')


output_path = '240517_C13M3_756.png'
plt.savefig(output_path)
plt.close()
