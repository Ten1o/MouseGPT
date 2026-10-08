import pandas as pd
import umap # pip install umap-learn
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime
from scipy.cluster.hierarchy import linkage, fcluster
import argparse
from pathlib import Path
from config import resolve_groups, reduction_paths

parser = argparse.ArgumentParser()
parser.add_argument('--n_neigh', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--group_config', nargs='+', required=True, help='Names defined in config.py; example selects only the published sample.')
parser.add_argument('--min_dist', type=float, default=0.1)
parser.add_argument('--workspace', type=str, default='workspace')
parser.add_argument('--output', type=str, default='output')


args = parser.parse_args()
n_neigh = args.n_neigh
model_name = args.model
workspace_folder = args.workspace
output_folder = args.output


groups = resolve_groups(args.group_config)
Path(output_folder).mkdir(parents=True, exist_ok=True)
embedding_path, linkage_path = reduction_paths(
    output_folder, args.group_config, 1, model_name, n_neigh, args.min_dist
)

# Choose the dimensionality reduction method
method = 'umap'  # Change to 'tsne' for t-SNE
part = 'des'


output_path = embedding_path
features_all = []
for group_name in tqdm(groups):
    with pd.HDFStore(f'{workspace_folder}/{model_name}/{group_name}.h5') as store:
        data = store.get(f'features_{part}')
        features_all.append(data)
data = np.vstack(features_all)

# Apply the chosen dimensionality reduction method
if method == 'umap':
    print('UMAP starts', datetime.now().time().strftime("%H:%M:%S"))
    start = time.time()
    reducer = umap.UMAP(n_components=2, random_state=42, verbose=True, n_neighbors=n_neigh, min_dist=args.min_dist)
    embedding = reducer.fit_transform(data)
    print(f'UMAP ends [{time.time() - start}s]')
    np.save(output_path, embedding)


output_path = linkage_path
low_dim_data_path = embedding_path
data = np.load(low_dim_data_path)
print('Clustering starts', datetime.now().time().strftime("%H:%M:%S"))
start = time.time()
linkage_matrix = linkage(data, method='single')
print(f'Clustering ends [{time.time() - start}s]')
np.save(output_path, linkage_matrix)
print(f'Saved to {output_path}.')
