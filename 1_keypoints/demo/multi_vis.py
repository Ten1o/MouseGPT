import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy
import os
from tqdm import tqdm
from functools import partial
from glob import glob

from scipy.signal import savgol_filter

import multiprocessing


colors = [[1.,         0.,         0.,         1       ],
            [1.         ,0.26086957 ,0.         ,1       ],
            [1.         ,0.52173913 ,0.         ,1       ],
            [1.         ,0.7826087  ,0.         ,1       ],
            [0.95652174 ,1.         ,0.         ,1       ],
            [0.69565217 ,1.         ,0.         ,1       ],
            [0.43478261 ,1.         ,0.         ,1       ],
            [0.17391304 ,1.         ,0.         ,1       ],
            [0.         ,1.         ,0.08695652 ,1       ],
            [0.         ,1.         ,0.34782609 ,1       ],
            [0.         ,1.         ,0.60869565 ,1       ],
            [0.         ,1.         ,0.86956522 ,1       ],
            [0.         ,0.86956522 ,1.         ,1       ],
            [0.         ,0.60869565 ,1.         ,1       ],
            [0.         ,0.34782609 ,1.         ,1       ],
            [0.         ,0.08695652 ,1.         ,1       ],
            [0.17391304 ,0.         ,1.         ,1       ],
            [0.43478261 ,0.         ,1.         ,1       ],
            [0.69565217 ,0.         ,1.         ,1       ],
            [0.95652174 ,0.         ,1.         ,1       ],
            [1.         ,0.         ,0.52173913 ,1       ],
            [1.         ,0.         ,0.26086957 ,1       ],
            [1.         ,0.         ,1.         ,1       ],
            [1.         ,0.         ,1.         ,1       ],
            [1.         ,0.         ,0.         ,1       ],
            [1.         ,0.         ,0.         ,1       ]]

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

def find_files(directory, file_name):

    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:

            if file == file_name:

                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths



def update(frame, pbar, ax, data_3d):
    ax.clear()
    points = data_3d[frame]
    frame += 1
    # print(f'\r{frame}', end='')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')

    for bone, color in zip(bones, colors):
        x = [points[bone[0]-1][0], points[bone[1]-1][0]]
        y = [points[bone[0]-1][1], points[bone[1]-1][1]]
        z = [points[bone[0]-1][2], points[bone[1]-1][2]]
        ax.plot(x, y, z, c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=0)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.set_title(f'Frame {frame}')

    pbar.update(1)

def read_pred_3dpts(pred_mat_path):
    mat_data = scipy.io.loadmat(pred_mat_path)
    mocap = mat_data['mocap']
    data_3d = []

    for joint in mocap[0, 0]:
        data_3d.append(joint)
    data_3d = np.array(data_3d)
    data_3d = np.transpose(data_3d, [1, 0, 2])

    print("Shape of data_3d", data_3d.shape)

    return data_3d

def filter(data_3d):
    frame_num = len(data_3d)
    for point in range(data_3d.shape[1]):
        indices = np.where((data_3d[:, point, :]==0).all(axis=1))[0]

        if len(indices) == 0:
            continue
        else:
            print(f'point {point} interpolating...')

        # missing point interpolation
        min_miss_id = -1
        miss_num = len(indices)
        for id in range(miss_num):
            if indices[id] < 1 or indices[id] == frame_num-1:
                continue

            if min_miss_id == -1:
                min_miss_id = indices[id]

            if id == miss_num-1 or indices[id] == frame_num - 2 or indices[id+1] != indices[id] + 1:
                data_3d[min_miss_id:indices[id]+1, point] = (data_3d[min_miss_id-1, point] + data_3d[indices[id]+1, point]) / 2
                min_miss_id = -1


    # Apply the Savitzky-Golay filter to the 3D keypoint data
    window_length = 15
    poly_order = 3

    # Ensure window length is odd
    if window_length % 2 == 0:
        window_length += 1

    filtered_data_3d = np.empty_like(data_3d)

    # Apply the filter for each dimension separately
    for dim in range(3):
        for point in range(data_3d.shape[1]):
            filtered_data_3d[:, point, dim] = savgol_filter(data_3d[:, point, dim], window_length, poly_order)
    return filtered_data_3d

def draw(group):
    print(group)
    npy_path = f'{group}/tri_new.npy'
    if os.path.exists(npy_path):
        print(npy_path)
        data_3d = np.load(npy_path)

        frame_num = data_3d.shape[0]
        if data_3d.shape[-1] == 3:
            data_3d = data_3d.reshape(frame_num,24,3)
        else:
            data_3d = data_3d[:, : , :, :3].reshape(frame_num,24,3)

        data_3d_ori = data_3d.copy()
        data_3d = filter(data_3d)

        np.save(f'{group}/tri_filter.npy', data_3d)
        data_3d_ori[:, 10:24, :] = data_3d[:, 10:24, :]
        np.save(f'{group}/tri_filter_limbs.npy', data_3d_ori)
    return


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    with tqdm(total=frame_num, desc="Processing", unit="frame") as pbar:


        ani = FuncAnimation(fig, partial(update, pbar=pbar, ax=ax, data_3d=data_3d), frames=frame_num, interval=1000/60)


        ani.save(f'{group}/tri_filter.mp4', writer='ffmpeg')

        plt.show()

    print(f'finish {group}')


if __name__ == '__main__':
    xmin, xmax, ymin, ymax, zmin, zmax = -0.02, 0.28, -0.02, 0.28, -0.02, 0.19

    pool = multiprocessing.Pool(processes=4)
    groups = []
    for date in tqdm(sorted(glob(os.environ['MOUSEGPT_DATE_GLOB']))):
        for group in sorted(glob(f'{date}/*')):
            groups.append(group)

    pool.map(draw, sorted(groups))
