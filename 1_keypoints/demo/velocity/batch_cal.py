import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as CM
from glob import glob

from scipy.ndimage import uniform_filter1d

for folder in [value.strip() for value in os.environ['MOUSEGPT_DATES'].split(',') if value.strip()]:
    for file in sorted(glob(os.environ['MOUSEGPT_DATA_ROOT'] + f'/{folder}/*/tri_new.npy')):
        print(file)
        # continue
        kpts = np.load(file)

        x = kpts[:, 5, 0]
        y = kpts[:, 5, 1]
        z = kpts[:, 5, 2]
        heatmap,xedge,yedge = np.histogram2d(x, y , bins=50)

        extent = [xedge[0], xedge[-1], yedge[0], yedge[-1]]
        plt.figure(figsize=(10,5))

        plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=CM.jet, interpolation='spline16')
        plt.colorbar(label='Frequency')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Heatmap of XY Trajectory')

        plt.savefig(f"output/{file.split('/')[-2]}_pos.png")

        pos = kpts[:, 5, :]
        movement = [pos[i]-pos[i-1] for i in range(1,len(pos))]
        speed = np.linalg.norm(np.array(movement), axis=1) * 60 * 100

        time = np.array(range(len(speed))) / 60 / 60

        windowsize = 200
        smooth_speed = uniform_filter1d(speed, size=windowsize)

        plt.figure(figsize=(10,5))
        plt.plot(time, speed, color='blue', linestyle='-', label='Original', alpha=0.4, linewidth=1)

        plt.plot(time, smooth_speed, color='red', linestyle='-', linewidth=2, label='Smoothed')
        plt.xlabel('Time(min)')
        plt.ylabel('Speed(mm/s')
        plt.xlim(0, None)
        plt.title('Speed Over Time')
        plt.legend()
        plt.savefig(f"output/{file.split('/')[-2]}_speed.png")
