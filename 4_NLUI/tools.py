from langchain_core.tools import tool
import os
from runtime_config import get_api_key

from tools_function.cluster import cluster as cluster_func
from tools_function.cluster_utils import export_cluster_labels
from tools_function.search import search as search_func
from tools_function.outlier import outlier as outlier_func
from tools_function.classification import classification as classification_func


from collections import Counter
from glob import glob
import pandas as pd
from datetime import datetime


tool_config = {
    "file_path": 'data/240419_C11M4.h5',
    "img_folder": 'data/240419_C11M4_imgs',
    "workspace": 'workspace/240419_C11M4'
}


@tool
def tool_classification(label: list) -> dict:
    """
    Description: This tool performs classification based on the user input of labels. The user specifies the category description/name to be clustered in the input. The results of the classification will generate a statistic to be returned.
    Use this tool if the user explicitly specifies a description or name to be categorised. If the user only says that clustering is needed and does not give the centre of the clusters, another tool is used: tool_cluster.

    Input param:
        - label: A list containing a series of strs, each of which describes a type of behaviour that the user has focused on. It can be a word, some phrase or a sentence.

    Output:
        - A dictionary with 1. the paths of the classification result, and 2. the statistics.
    """
    print(f'\033[95m')
    print("label:", label)
    print(f'\033[0m')
    
    h5_file_path, img_folder, api_key = tool_config["file_path"], tool_config["img_folder"], get_api_key("EMBEDDING")

    img_list = sorted(glob(os.path.join(img_folder, "*.jpg")))

    classification_result = classification_func(h5_file_path, label, api_key)
    save_path = os.path.join(tool_config["workspace"], "classification_" + datetime.now().strftime("%y%m%d_%H%M%S_%f") + ".csv")
    static = export_cluster_labels(
        list(classification_result), list(classification_result.values()), label, save_path,
    )

    result_params = {
        "static": static,
        "save_path": save_path
    }
    return result_params



@tool
def tool_cluster() -> dict:
    """
    Description: This tool clusters and saves the results. The results of the clustering will generate a statistic to be returned.
    If the user does not explicitly specify the description or name of the classification that needs to be performed, but only says that clustering or clustering into several categories is performed, use this tool. If the user gives a specific description of the clustering, use another tool: tool_classification.

    Output:
        - A dictionary with 1. the paths of the clustering result, and 2. the statistics of the clustering results.
    """
    print(f'\033[95m')
    print("cluster")
    print(f'\033[0m')
    
    h5_file_path, img_folder, api_key = tool_config["file_path"], tool_config["img_folder"], get_api_key("ANALYSIS")

    img_list = sorted(glob(os.path.join(img_folder, "*.jpg")))

    labels_des, behaviors = cluster_func(h5_file_path, img_folder, api_key)
    with pd.HDFStore(h5_file_path, mode="r") as store:
        frame_ids = [str(value).zfill(8) for value in store.get("features_des").index]
    save_path = os.path.join(tool_config["workspace"], "cluster.csv")
    static = export_cluster_labels(frame_ids, labels_des, behaviors, save_path)

    result_params = {
        "static": static,
        "save_path": save_path
    }
    return result_params

@tool
def tool_behavior(frame_idx: int) -> dict:
    """
    A behaviour analyzer for mice.

    Input param:
        - frame_idx: the current frame index

    Based on the input frame index, this tool allows for a precise behavioural description of mice. A dictionary is returned, which is divided into two parts: 1. the description part gives an exhaustive behavioural description of the mice. 2. the keywords part gives a keyword summary of the mice's behaviour.
    """

    print(f'\033[95m')
    print("frame_idx:", frame_idx)
    print(f'\033[0m')

    h5_file_path, img_folder = tool_config["file_path"], tool_config["img_folder"]

    # h5_file_path = tool_config["file_path"]
    frame_idx_str = str(frame_idx).zfill(8)
    with pd.HDFStore(h5_file_path, mode='r') as store:
        data = store.get('descriptions')
        matches = data.loc[data['id'].astype(str).str.zfill(8) == frame_idx_str, 'description']
        if matches.empty:
            raise ValueError(f'No stored description for frame {frame_idx_str}.')
        description = matches.iloc[0]

    return description



@tool
def tool_search(behavior_description: str) -> dict:
    """
    A search tool for behaviour. Based on the behaviour description entered by the user, search for the frame index corresponding to the closest behaviour description among all the data. return the five closest frame indexes, and sort and save the closeness of all the frames.

    Input param:
        - behavior_description: Description of target behaviour

    Output: 
        - a dict
        {
            "top_5": a list with 5 frame index
            "save_path": str, the paths of the searching result
        }
    """

    print(f'\033[95m')
    print("behavior_description:", behavior_description)
    print(f'\033[0m')


    h5_file_path, img_folder, api_key = tool_config["file_path"], tool_config["img_folder"], get_api_key("EMBEDDING")
    img_list = sorted(glob(os.path.join(img_folder, "*.jpg")))

    sim_index, similarity = search_func(h5_file_path, behavior_description, api_key)



    save_path = os.path.join(tool_config["workspace"], "search_" + datetime.now().strftime("%y%m%d_%H%M%S_%f") + ".csv")
    # os.makedirs(save_path, exist_ok=True)


    with open(save_path, "w") as file:
        file.write(f"frame_id,similarity\n")

        for i, item in enumerate(similarity):
            file.write(f"{sim_index[i]},{similarity[i]}\n")


    top_5 = sim_index[0:5]

    result_params = {
        "top_5": top_5,
        "save_path": save_path
    }
    return result_params



@tool
def tool_outlier() -> dict:
    """
    A outlier detection tool for behaviour. Detect the frame index corresponding to the least frequently occurring behaviours, actions or bursts. Return the five least frequently occuring frame indexes, and sort and save for all the frames.

    Output: 
        - a dict
        {
            "top_5": a list with 5 frame index
            "save_path": str, the paths of the outlier detection result
        }
    """

    print(f'\033[95m')
    print("outlier detection")
    print(f'\033[0m')


    h5_file_path, img_folder = tool_config["file_path"], tool_config["img_folder"]
    img_list = sorted(glob(os.path.join(img_folder, "*.jpg")))

    outlier_index, outlier_degree = outlier_func(h5_file_path)



    save_path = os.path.join(tool_config["workspace"], "outlier.csv")
    # os.makedirs(save_path, exist_ok=True)


    with open(save_path, "w") as file:
        file.write(f"frame_id,outlier_degree\n")

        for i, item in enumerate(outlier_degree):
            file.write(f"{outlier_index[i]},{outlier_degree[i]}\n")


    top_5 = outlier_index[0:5]

    result_params = {
        "top_5": top_5,
        "save_path": save_path
    }
    return result_params
