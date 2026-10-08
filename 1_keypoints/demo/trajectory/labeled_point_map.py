import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as CM
from glob import glob
import pandas as pd
from argparse import ArgumentParser
import os
from tqdm import tqdm

def load_data(group, h5_folder, npy_folder, name='cluster'):
    """Load categories and labels from HDF5 file."""
    h5_file_path = os.path.join(h5_folder, f"{group}.h5")
    npy_file_path = os.path.join(npy_folder, f"{group}.npy")
    kpts_3d = np.load(npy_file_path)

    with pd.HDFStore(h5_file_path) as store:
        # cates = store.get(f'category_{part_name}_keyword')
        # label_keyword = store.get(f'label_{part_name}_keyword')
        if name == 'cluster':
            cates = store.get(f'category_All_cluster')
            label_keyword = store.get(f'label_All_cluster')
            labels = np.array(label_keyword[1]).reshape(-1)

        ids = list(label_keyword.index)
    return cates, labels, ids, kpts_3d

def draw_point_trajectory(labels, cates, kpts, save_path):
    """
    Plot a 2D xy trajectory from 3D points, colored by label.

    Args:
        labels: One-dimensional array of point labels in the range 0 to 3.
        kpts: Array of shape (N, 3) containing the 3D point coordinates.
    """

    colors = ['red', 'green', 'blue', 'purple']


    plt.figure(figsize=(8, 8))


    for label in range(4):

        idx = labels == label

        x = kpts[idx, 5, 0]
        y = kpts[idx, 5, 1]

        plt.scatter(x, y, c=colors[label], label=cates[0][label], alpha=0.5, s=10, edgecolors='none')


    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Point Trajectory with Labels')
    plt.legend()


    plt.savefig(save_path)
    plt.close()

data = {'medicine': [
        '240517_C13M1', '240517_C13M2', '240517_C13M3', '240517_C13M4', '240517_C13M5',
        '240517_C14M1', '240518_C14M2', '240518_C14M3', '240518_C15M1', '240518_C15M2',
        '240519_C17M1', '240519_C17M2', '240519_C17M3', '240519_C17M4', '240519_C17M5'
    ],
    'extended_medicine': [
        '231116_C4M1', '231116_C4M2', '231116_C4M3', '231116_C4M4', '231116_C4M5',
        '231116_C4M6', '240202_C5M1', '240202_C5M2', '240202_C5M3', '240202_C5M4',
        '240202_C5M5', '231115_C3M1', '231115_C3M2', '231115_C3M3', '231115_C3M4',
        '231115_C3M5', '231115_C3M6'
    ],
    'depression_new_all' : [
        "240419_C11M4", "240420_C10M1", "240420_C10M2", "240420_C10M4", "240420_C12M1",
        "240420_C12M2", "240420_C12M4", "240420_C9M1", "240420_C9M2", "240420_C9M3",
        "240420_C9M4", "240419_C11M1", "240419_C11M2", "240419_C11M3", "240420_C10M3",
        "240420_C12M3", "240421_C7M2", "240421_C7M3", "240421_C7M4", "240421_C8M1",
        "240421_C8M2", "240421_C8M4", "240512_C11M4", "240512_C12M1", "240512_C12M2",
        "240512_C12M4", "240513_C10M1", "240513_C10M2", "240513_C10M4", "240513_C9M1",
        "240513_C9M2", "240513_C9M3", "240513_C9M4", "240507_C7M1", "240507_C7M2",
        "240507_C7M3", "240507_C7M4", "240507_C8M1", "240507_C8M2", "240512_C11M1",
        "240512_C11M2", "240512_C11M3", "240513_C10M3"
    ]}


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='internvl2-full-18000',
        choices=['internvl2-full-18000', 'minicpm-2.6-full-fp32-45900'])
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Directory for trajectory figures.')
    parser.add_argument('--h5-folder', required=True, help='HDF5 folder for the selected model.')
    parser.add_argument('--npy-folder', required=True, help='Folder containing session keypoint arrays.')

    args = parser.parse_args()

    h5_folder = args.h5_folder
    npy_folder = args.npy_folder

    for k, v in data.items():
        print(k)
        out_folder = os.path.join(args.output_path, args.model,k)
        os.makedirs(out_folder, exist_ok=True)
        for group in tqdm(v):
            save_path = os.path.join(out_folder, f'{group}.png')
            cates, labels, ids, kpts_3d = load_data(group, h5_folder, npy_folder)
            kpts = kpts_3d[list(map(int, ids)),:,:2]
            draw_point_trajectory(labels, cates, kpts, save_path)
