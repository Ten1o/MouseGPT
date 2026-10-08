import pandas as pd
import json
from scipy.cluster.hierarchy import ClusterNode, to_tree
from tools_function.cluster_utils import CLUSTER_CACHE_VERSION, bfs_traverse_print, save_des_labels
from runtime_config import validate_frame_ids
import os





def cluster(h5_file_path, img_folder, api_key):



    system_prompt = 'You will be presented serveral text samples from cluster A, and several text samples from cluster B. You should decide whether or not to merge cluster A and B. All the samples are from bio-science experiments to describe the behavior of the mouse. \
    \
    You should decide whether to merge cluster A and B based on the behavioral similarity in the given sampled text between cluster A and cluster B. And give a score from 0 to 1 (1 is the most similar, 0 is not similar at all) to tell how similar of two clusters. For behaviors with major difference, such as rearing and grooming, they should have low behavioral similarity scores.\
    \
    You should also summarize samples from cluster A, and cluster B, separately, using one keyword based on the mouse behavior, as concise as possible. The summarized keywords should be the most important, most distinguishable behaviors, and they should be in common for the most samples. \
    \
    You should avoid fuzzy terms such as "alert", "calm", etc. that does not reflect direct behavior. The good summarization reflects distinguishable behavior directly, such as "walking", "running", "grooming", or "rearing", etc. You may also summarize fine-grained level behaviors, such as limb motion, or other body parts, if they did have some distinguishable features in common. Please use only one word to summarize the unique keyword. \
    \
    The output should be a json format, the keys are "similarity_score": The similarity scores between behavior clusters, range from 0-1 ; "reason": reason for decision of merge or not; "merge": "True" or "False". "cluster_A": the summarized keywords for samples from cluster A; "cluster_B": the summarized keyword for samples from cluster B;'


    with pd.HDFStore(h5_file_path, mode='r') as store:
        data = store.get('features_des')
        descriptions = store.get('descriptions')
        frame_ids = validate_frame_ids(descriptions['id'], data.index)
        description_content_list = descriptions['description'].tolist()
        linkage_matrix = store.get('linkage_matrix')
        img_paths_list = [f'{img_folder}/{frame_id}.jpg' for frame_id in frame_ids]

        if len(linkage_matrix) != len(frame_ids) - 1:
            raise ValueError("The linkage matrix must describe every stored feature row.")
        root = ClusterNode(0) if len(frame_ids) == 1 else to_tree(linkage_matrix)
        if root.count != len(frame_ids):
            raise ValueError("The clustering tree has a different number of leaves than stored rows.")
        from pathlib import Path
        import hashlib
        source = Path(h5_file_path).resolve()
        identity = f'{CLUSTER_CACHE_VERSION}:{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{Path(img_folder).resolve()}:{system_prompt}'
        cache_key = hashlib.sha256(identity.encode()).hexdigest()[:16]
        cache_dir = Path(__file__).resolve().parents[1] / 'workspace' / 'cluster_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        gpt_cluster_result_path = cache_dir / f'{cache_key}.json'
        if os.path.exists(gpt_cluster_result_path):
            with open(gpt_cluster_result_path, 'r') as json_file:
                gpt_cluster_result = json.load(json_file)
        else:
            gpt_cluster_result = bfs_traverse_print(root, system_prompt, api_key, description_content_list, img_paths_list, max_sample_num=50)
            save_des_labels(gpt_cluster_result, len(description_content_list))
            with open(gpt_cluster_result_path, 'w') as json_file:
                json.dump(gpt_cluster_result, json_file, indent=4)
        labels_des, behaviors = save_des_labels(gpt_cluster_result, len(description_content_list))
        return labels_des, behaviors


if __name__ == '__main__':
    print(cluster('data/240419_C11M4.h5', 'data/240419_C11M4_imgs', os.environ['OPENAI_API_KEY']))
