import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
from matplotlib.animation import FuncAnimation
from scipy.signal import savgol_filter
    
color = ['r', 'g']

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

xmin, xmax, ymin, ymax, zmin, zmax = -0.02, 0.28, -0.02, 0.28, -0.02, 0.19

def filter(data_3d):
    # Apply the Savitzky-Golay filter to the 3D keypoint data
    window_length = 15
    poly_order = 5

    # Ensure window length is odd
    if window_length % 2 == 0:
        window_length += 1

    filtered_data_3d = np.empty_like(data_3d)

    # Apply the filter for each dimension separately
    for mouse in range(2):
        for dim in range(3):
            for point in range(data_3d.shape[2]):
                filtered_data_3d[:, mouse, point, dim] = savgol_filter(data_3d[:, mouse, point, dim], window_length, poly_order)
    return filtered_data_3d

def draw_3d_kpts(ax, points, id, frame_id):

    ax.scatter(points[:,0], points[:,1], points[:,2], c=color[id], marker='o')

    for bone in bones:
        x = [points[bone[0]-1][0], points[bone[1]-1][0]]
        y = [points[bone[0]-1][1], points[bone[1]-1][1]]
        z = [points[bone[0]-1][2], points[bone[1]-1][2]]
        ax.plot(x, y, z, c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title(f'{frame_id}')

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax.view_init(elev=10, azim=50)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

data_3D = np.load('multi_mouse_free_2.npy')

frame_num = len(data_3D)
indices = np.where((data_3D==0).all(axis=(1,2,3)))[0]

last_frame = []
last_id = [0,1]

id_list = []

for frame_id in tqdm(range(frame_num)):
    frame = data_3D[frame_id]

    if frame_id not in indices:

        if frame_id != 0:
            diff0 = np.linalg.norm(frame[0] - last_frame, axis=(1,2))
            diff1 = np.linalg.norm(frame[1] - last_frame, axis=(1,2))
            min_diff_pair = np.argmin(np.array([diff0, diff1]))
            if min_diff_pair % 3 != 0:
                last_id.reverse()
            
        last_frame = frame
        
    id_list.append(last_id.copy())
    
id_list = np.array(id_list)
data_3D_range = range(frame_num)
tracked_data_3D = np.stack([data_3D[data_3D_range, id_list[:,0],:,:], data_3D[data_3D_range, id_list[:,1],:,:]], axis=1)

# missing point interpolation
min_miss_id = -1
miss_num = len(indices)
for id in range(miss_num):
    if indices[id] < 1 or indices[id] == frame_num-1:
        continue
    
    if min_miss_id == -1:
        min_miss_id = indices[id]

    if id == miss_num-1 or indices[id] == frame_num - 2 or indices[id+1] != indices[id] + 1:
        tracked_data_3D[min_miss_id:indices[id]+1] = (tracked_data_3D[min_miss_id-1] + tracked_data_3D[indices[id]+1]) / 2
        min_miss_id = -1

filtered_data_3D = filter(tracked_data_3D)


def update(frame_id):
    print(frame_id, end='\r')
    global last_frame, last_id, data_3D, ax, id_list

    frame = filtered_data_3D[frame_id]

    ax.clear()
    for mouse, id in zip(frame, [0,1]):
        draw_3d_kpts(ax, mouse, id, frame_id)

    
ani = FuncAnimation(fig, update, frames=range(39400, frame_num), interval=1000/60)


ani.save('animation-all-free-2-track-filter-inter.mp4', writer='ffmpeg')


plt.show()