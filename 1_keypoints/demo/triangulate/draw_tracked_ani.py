import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
from matplotlib.animation import FuncAnimation
    
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

def update(frame_id):
    print(frame_id, end='\r')
    global last_frame, last_id, data_3D, ax

    frame = data_3D[frame_id]

    if frame_id not in indices:

        if frame_id != 0:
            diff0 = np.linalg.norm(frame[0] - last_frame, axis=(1,2))
            diff1 = np.linalg.norm(frame[1] - last_frame, axis=(1,2))
            min_diff_pair = np.argmin(np.array([diff0, diff1]))
            if min_diff_pair % 3 != 0:
                last_id.reverse()
            
        last_frame = frame

    ax.clear()
    for mouse, id in zip(frame, last_id):
        draw_3d_kpts(ax, mouse, id, frame_id)

    
ani = FuncAnimation(fig, update, frames=range(frame_num), interval=1000/60)


ani.save('animation-all-4-track.mp4', writer='ffmpeg')


plt.show()