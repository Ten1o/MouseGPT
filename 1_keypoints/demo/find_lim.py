from glob import glob
import os
import numpy as np
from tqdm import tqdm

xmin, xmax, ymin, ymax, zmin, zmax = None,None,None,None,None,None

for date in tqdm(sorted(glob(os.environ['MOUSEGPT_DATE_GLOB']))):
    for group in sorted(glob(f'{date}/*')):
        print(group)
        npy_path = f'{group}/tri_new.npy'
        if os.path.exists(npy_path):
            data_3d = np.load(npy_path)

            frame_num = data_3d.shape[0]
            if data_3d.shape[-1] == 3:
                data_3d = data_3d.reshape(frame_num,24,3)
            else:
                data_3d = data_3d[:, : , :, :3].reshape(frame_num,24,3)

            valid_indices = np.isfinite(data_3d[:, :, 0]) & np.isfinite(data_3d[:, :, 1]) & np.isfinite(data_3d[:, :, 2])


            filtered_data_3d = data_3d[valid_indices]

            xmin_t, xmax_t = np.min(filtered_data_3d[:, 0]), np.max(filtered_data_3d[:, 0])
            ymin_t, ymax_t = np.min(filtered_data_3d[:, 1]), np.max(filtered_data_3d[:, 1])
            zmin_t, zmax_t = np.min(filtered_data_3d[:, 2]), np.max(filtered_data_3d[:, 2])
            print(xmin_t, xmax_t, ymin_t, ymax_t, zmin_t, zmax_t)
            if xmin is None:
                xmin, xmax, ymin, ymax, zmin, zmax = xmin_t, xmax_t, ymin_t, ymax_t, zmin_t, zmax_t
            else:
                xmin = min(xmin, xmin_t)
                ymin = min(ymin, ymin_t)
                zmin = min(zmin, zmin_t)
                xmax = max(xmax, xmax_t)
                ymax = max(ymax, ymax_t)
                zmax = max(zmax, zmax_t)

print(xmin, xmax, ymin, ymax, zmin, zmax)
