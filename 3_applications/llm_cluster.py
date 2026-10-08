import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import fcluster
import requests
import json
import re
from tqdm import tqdm

from config import resolve_groups, reduction_paths

import argparse

from llm_utils import parse_json_object
from pathlib import Path



system_prompt = "You are a helpful assistant that analyzes clusters based on provided keywords."


def load_data(group_files, img_workspace, downscale=1):
    if downscale != 1:
        raise ValueError("Use downscale=1 to retain the session frame index.")
    features_all, descriptions, images_a, images_b, group_names = [], [], [], [], []
    for group_file in tqdm(group_files):
        with pd.HDFStore(group_file, mode="r") as store:
            description_table = store.get("descriptions")
            features = store.get("features_des")
        ids = list(description_table["id"].map(str))
        if len(set(ids)) != len(ids) or ids != list(features.index.map(str)):
            raise ValueError(f"Description and embedding IDs are not aligned in {group_file}.")
        values = list(description_table["description"])
        if any(not isinstance(value, dict) or not isinstance(value.get("Keywords"), str) or not value["Keywords"].strip() for value in values):
            raise ValueError(f"Coarse clustering requires a nonempty Keywords field for each frame in {group_file}.")
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError(f"Nonfinite features in {group_file}.")
        features_all.append(features.to_numpy())
        descriptions.extend(values)
        group = Path(group_file).stem
        group_names.extend([group] * len(ids))
        images_a.extend(str(Path(img_workspace) / group / "pre_images/data3" / f"{frame_id}.jpg") for frame_id in ids)
        images_b.extend(str(Path(img_workspace) / group / "pre_images/data5" / f"{frame_id}.jpg") for frame_id in ids)
    if not features_all:
        raise ValueError("Select at least one session.")
    return np.vstack(features_all), descriptions, images_a, images_b, group_names


# Function to load group data
def load_groups(model_name, min_dist, downscale, n_neigh, group_config, *, workspace, npy_folder, img_workspace):
    groups = resolve_groups(group_config)

    print(f"Selected group configurations: {group_config}")
    print(groups)

    # Load data (assuming this function loads features, descriptions, and image paths)
    group_files = [f'{workspace}/{model_name}/{group}.h5' for group in groups]
    features, descriptions, image_paths_A, image_paths_B, group_names = load_data(group_files, img_workspace=img_workspace, downscale=downscale)
    description_df = pd.DataFrame(descriptions)

    low_dim_data_path, linkage_matrix_path = reduction_paths(
        npy_folder, group_config, downscale, model_name, n_neigh, min_dist
    )

    title = os.path.basename(low_dim_data_path)
    Z = np.load(linkage_matrix_path)
    low_dim_data = np.load(low_dim_data_path)
    print("Z.shape", Z.shape, "low_dim_data.shape", low_dim_data.shape)

    return features, descriptions, image_paths_A, image_paths_B, description_df, low_dim_data, Z, title, group_names


# Function to determine cluster merges using GPT
def determine_cluster_merges(most_common_words, new_cluster_names_dict):
    prompt = """
    Based on the following most common keywords for each cluster, determine which clusters should be merged.
    Please pay special attention to behaviors like standing, jumping, grooming, walking, stationary, sniffing, and running. Try not to use '/'. Do not mix standing and staionary as one cluster. 
    1. !!!!!!!!!!!!!!!!! NOTE: IF "front paws ON THE GROUND", and "STANDING", and "STATIONARY" OCCUR simultaneously, IT SHOULD BE "STATIONARY" !!!!!!!!!!!!!!!!!
    2. NOTE: IF "front paws RAISED" and "staionary" OCCUR, IT SHOULD BE "STANDING".
    3. NOTE: IF "SITTING" and "staionary" OCCUR, IT SHOULD BE "SITTING" .
    4. NOTE: ONCE "grooming" OCCUR, IT SHOULD BE "grooming".
    even if they are not the most common keywords. These behaviors should not be ignored when making your decisions. The new clusters should be as diverse as possible, no less than 4. 

    You should not merge clusters with label "-1", just keep it. 
    
    Output the results in the following JSON format:
    {
        "New Cluster Names": {
            "-1(int)": "Others",
            "0(int)": "Behavior Summary 1",
            "1(int)": "Behavior Summary 2",
            ...
        },
        "Mapping": {
            "Old Cluster ID 1(int)": "New Cluster ID 1(int)",
            "Old Cluster ID 2(int)": "New Cluster ID 2(int)",
            ...
        }
    }

    Here are the most common keywords for each cluster in the format {cluster_id}: {keywords1}, {keywords2}, ...:
    """
    for cluster, words in most_common_words.items():
        prompt += f"{cluster}: {', '.join(words)}\n"
    
    prompt += """
    Here are existing "New Cluster Names", you can add new distinct behavior to this dict when necessary
    """
    
    prompt += str(new_cluster_names_dict)
    
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        timeout=60
    )

    return str(completion.choices[0].message.content).strip()


# Function to update cluster labels based on GPT's response
def integer_id(value):
    if isinstance(value, bool):
        raise ValueError("Boolean values are not cluster IDs.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    raise ValueError(f"Invalid cluster ID: {value!r}")


def validate_merges(response, expected_ids, existing_names):
    names, mapping = response.get("New Cluster Names"), response.get("Mapping")
    if not isinstance(names, dict) or not isinstance(mapping, dict):
        raise ValueError("Merge response must contain name and mapping objects.")
    names = {str(integer_id(key)): name for key, name in names.items()}
    if any(integer_id(key) < -1 or not isinstance(name, str) or not name.strip() for key, name in names.items()):
        raise ValueError("Cluster names must be nonempty strings with IDs at least -1.")
    for key, name in existing_names.items():
        if str(key) in names and names[str(key)] != name:
            raise ValueError("A merge response cannot rename an earlier cluster.")
    all_names = {**existing_names, **names}
    mapping = {str(integer_id(key)): str(integer_id(value)) for key, value in mapping.items()}
    if set(mapping) != {str(integer_id(value)) for value in expected_ids}:
        raise ValueError("Merge mapping must cover exactly the supplied cluster IDs.")
    if any(value not in all_names for value in mapping.values()):
        raise ValueError("Every mapped cluster must have a name.")
    if "-1" in mapping and mapping["-1"] != "-1":
        raise ValueError("The unassigned cluster must retain ID -1.")
    return {"New Cluster Names": all_names, "Mapping": mapping}


def request_cluster_merges(words, existing_names, *, max_attempts=3):
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    error = None
    for _ in range(max_attempts):
        try:
            response = parse_json_object(determine_cluster_merges(words, existing_names))
            return validate_merges(response, words.keys(), existing_names)
        except Exception as exc:
            error = exc
    raise RuntimeError(f"No valid cluster mapping after {max_attempts} attempts.") from error


def update_cluster_labels(labels, merge_results):
    merge_results = validate_merges(merge_results, np.unique(labels), {})
    mapping = {int(old): int(new) for old, new in merge_results["Mapping"].items()}
    new_labels = np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)
    names = {int(key): value for key, value in merge_results["New Cluster Names"].items()}
    return new_labels, names


# Function to find the most common words in each cluster
def get_most_common_words(labels, description_df):
    stop_words = ['tail extended', 'hind paws on ground']
    most_common_words = {}
    description_df['cluster'] = labels
    
    for cluster in np.unique(labels):
        cluster_descriptions = description_df[description_df['cluster'] == cluster]['Keywords']
        all_words = []
        for desc in cluster_descriptions:
            if isinstance(desc, str):
                behaviors = desc.split(', ')
                # Remove stop words
                behaviors = [word for word in behaviors if word not in stop_words]
                all_words.extend(behaviors)
        most_common_word = Counter(all_words).most_common(5)
        most_common_word = [word for word, freq in most_common_word]
        most_common_word.insert(0, f'[{len(cluster_descriptions)}]')
        most_common_words[cluster] = most_common_word
        
    return most_common_words



# Function to count frames for each group and behavior
def count_frames_per_group(labels, group_names, summary_labels):
    # Create a dictionary to store counts for each group and behavior
    group_behavior_counts = {}

    # Replace behavior labels with behavior names using summary_labels
    behavior_names = [summary_labels[label] for label in labels]
    
    # Iterate over all unique group names
    for group in set(group_names):
        # Filter behavior names for the current group
        group_behaviors = np.array(behavior_names)[np.array(group_names) == group]
        
        # Count the occurrences of each behavior (name) in this group
        behavior_counts = Counter(group_behaviors)
        
        # Store the counts in the dictionary
        group_behavior_counts[group] = behavior_counts
    
    # Convert the dictionary to a DataFrame and transpose
    group_behavior_df = pd.DataFrame(group_behavior_counts).fillna(0).astype(int).T
    group_behavior_df.index.name = 'Group_Name'
    return group_behavior_df


def filter_minor_cluster(Z, max_d, thresh=3):
    n_frames = len(Z) + 1
    if max_d < 1 or thresh < 1 or max_d * thresh > n_frames:
        raise ValueError("Requested cluster count and minimum size cannot fit the available frames.")
    for cluster_count in range(max_d, n_frames + 1):
        labels = fcluster(Z, t=cluster_count, criterion="maxclust").astype(np.int64) - 1
        unique, counts = np.unique(labels, return_counts=True)
        for cluster, count in zip(unique, counts):
            if count < thresh:
                labels[labels == cluster] = -1
        if len(np.unique(labels[labels != -1])) >= max_d:
            return labels
    raise ValueError("Could not find the requested number of clusters at the minimum size.")


def category_tables(labels, summary_labels, ids):
    labels = np.asarray(labels)
    if len(labels) != len(ids):
        raise ValueError("Each frame must have exactly one cluster label.")
    order = sorted(summary_labels)
    if not set(labels).issubset(order):
        raise ValueError("Each cluster label must have a behavior name.")
    positions = {label: index for index, label in enumerate(order)}
    categories = pd.DataFrame([summary_labels[label] for label in order])
    assigned = pd.DataFrame([positions[label] for label in labels], index=list(ids))
    return categories, assigned


def main(args):
    global client
    if args.downscale != 1:
        raise ValueError("Use --downscale 1 for per-frame HDF5 output.")
    if args.max_attempts < 1:
        raise ValueError("--max_attempts must be positive.")
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=60)
    features, descriptions, images_a, images_b, description_df, projection, Z, title, group_names = load_groups(
        args.model_name, args.min_dist, args.downscale, args.n_neigh, args.group_config,
        workspace=args.workspace, npy_folder=args.npy_folder, img_workspace=args.img_workspace,
    )
    if len(features) != len(Z) + 1 or len(projection) != len(features):
        raise ValueError("Clustering inputs and session frame counts differ.")
    labels = filter_minor_cluster(Z, args.n_clusters, args.min_cluster_size)
    words = get_most_common_words(labels, description_df)
    merges = {"New Cluster Names": {}, "Mapping": {}}
    items = list(words.items())
    for start in range(0, len(items), 60):
        chunk = dict(items[start:start + 60])
        result = request_cluster_merges(chunk, merges["New Cluster Names"], max_attempts=args.max_attempts)
        merges["New Cluster Names"].update(result["New Cluster Names"])
        merges["Mapping"].update(result["Mapping"])
    labels, names = update_cluster_labels(labels, merges)
    output = Path(args.ofolder)
    output.mkdir(parents=True, exist_ok=True)
    groups = resolve_groups(args.group_config)
    groups_array = np.asarray(group_names)
    prepared = []
    counts = []
    for group in groups:
        h5_file = Path(args.workspace) / args.model_name / f"{group}.h5"
        group_labels = labels[groups_array == group]
        with pd.HDFStore(h5_file, mode="r") as store:
            ids = store.get("descriptions")["id"]
        categories, assigned = category_tables(group_labels, names, ids)
        prepared.append((h5_file, categories, assigned))
        counts.append({"group": group, **dict(Counter(names[int(label)] for label in group_labels))})
    for h5_file, categories, assigned in prepared:
        with pd.HDFStore(h5_file) as store:
            store.put(f"category_All_cluster_{args.date}", categories)
            store.put(f"label_All_cluster_{args.date}", assigned)
    pd.DataFrame(counts).fillna(0).to_excel(output / f"{args.model_name}_{args.group_config[0]}_All_cluster.xlsx", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster and name behavior descriptions.")
    for name in ("model_name", "npy_folder", "workspace", "img_workspace", "ofolder", "date"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--n_neigh", type=int, required=True)
    parser.add_argument("--min_dist", type=float, required=True)
    parser.add_argument("--group_config", nargs="+", required=True)
    parser.add_argument("--downscale", type=int, default=1)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--n_clusters", type=int, default=400)
    parser.add_argument("--min_cluster_size", type=int, default=3)
    main(parser.parse_args())
