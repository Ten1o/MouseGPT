"""Language-guided tree traversal with explicit coverage and label validation."""
from collections import deque
import csv
import json
import math
from numbers import Integral
from pathlib import Path
import random

import numpy as np

from tools_function.provider import post_json

UNASSIGNED = -1
UNASSIGNED_NAME = "Unassigned"
CLUSTER_CACHE_VERSION = "2"


def parse_merge(value):
    """Accept JSON booleans and the equivalent legacy string representation."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError("merge must be a JSON boolean or a 'true'/'false' string.")


def extract_message_content(data):
    """Read an object from plain or fenced assistant JSON, including nested braces."""
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("The provider did not return a text decision.")
    decoder = json.JSONDecoder()
    for offset, character in enumerate(content):
        if character != "{":
            continue
        try:
            decision, _ = decoder.raw_decode(content[offset:])
        except ValueError:
            continue
        if isinstance(decision, dict):
            return decision
    raise ValueError("The provider did not return a JSON decision.")


def parse_decision(result):
    message = extract_message_content(result)
    message["merge"] = parse_merge(message["merge"])
    score = message["similarity_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("similarity_score must be a finite number between zero and one.")
    for key in ("cluster_A", "cluster_B"):
        if not isinstance(message[key], str) or not message[key].strip():
            raise ValueError(f"{key} must be a nonempty behavior label.")
    return message


def gpt_request(system_prompt, A_samples, B_samples, api):
    prompt = f"Samples from Cluster A: {A_samples}\nSamples from Cluster B: {B_samples}"
    result = post_json(
        "https://api.openai.com/v1/chat/completions",
        api_key=api,
        payload={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "top_p": 0.5,
            "max_tokens": 2000,
        },
        validate=parse_decision,
    )
    result["system_prompt"] = system_prompt
    result["prompt"] = prompt
    return result


def get_cluster_elements(node):
    """Return leaf positions without a recursion-depth limit on unbalanced trees."""
    leaves = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.is_leaf():
            leaves.append(current.id)
        else:
            stack.extend((current.right, current.left))
    return leaves


def bfs_traverse_print(root, system_prompt, api, description_content_list, img_paths_list, max_sample_num=200, max_level=8):
    """Record all terminal groups; leaves and depth-limited groups stay unassigned."""
    descriptions = list(description_content_list)
    if max_sample_num < 1 or max_level < 0:
        raise ValueError("Sampling must be positive and the maximum level nonnegative.")
    if len(descriptions) != len(img_paths_list) or sorted(get_cluster_elements(root)) != list(range(len(descriptions))):
        raise ValueError("Tree leaves, descriptions and image paths must refer to the same rows.")
    decisions = {}
    decision_count = 0
    queue = deque([(root, 0, f"Cluster {root.id}")])
    while queue:
        node, level, cluster_name = queue.popleft()
        if node.is_leaf() or level > max_level:
            decisions[node.id] = {
                "level": level,
                "cluster_curr": cluster_name,
                "terminal_reason": "leaf" if node.is_leaf() else "depth_limit",
                "message": {"merge": False, "ids": get_cluster_elements(node)},
            }
            continue
        left_ids = get_cluster_elements(node.left)
        right_ids = get_cluster_elements(node.right)
        left_sampled = random.sample(left_ids, min(max_sample_num, len(left_ids)))
        right_sampled = random.sample(right_ids, min(max_sample_num, len(right_ids)))
        result = gpt_request(
            system_prompt,
            [descriptions[index] for index in left_sampled],
            [descriptions[index] for index in right_sampled],
            api,
        )
        message = parse_decision(result)
        if message["merge"]:
            message["ids"] = left_ids + right_ids
        decisions[node.id] = {
            "level": level,
            "message": message,
            "cluster_curr": cluster_name,
            "left_img_paths": [[img_paths_list[index] for index in left_sampled]],
            "right_img_paths": [[img_paths_list[index] for index in right_sampled]],
        }
        if not message["merge"]:
            queue.append((node.left, level + 1, message["cluster_A"]))
            queue.append((node.right, level + 1, message["cluster_B"]))
        decision_count += 1
    print(f"GPT made {decision_count} clustering decisions.")
    return decisions


def save_des_labels(gpt_cluster_result, sample_num):
    """Assign only confirmed merged groups; -1 represents every unassigned frame."""
    if isinstance(sample_num, bool) or not isinstance(sample_num, Integral) or sample_num < 0:
        raise ValueError("sample_num must be a nonnegative integer.")
    labels = np.full(sample_num, UNASSIGNED, dtype=np.int64)
    covered = np.zeros(sample_num, dtype=bool)
    behaviors = []
    for entry in gpt_cluster_result.values():
        message = entry["message"]
        merged = parse_merge(message["merge"])
        if "ids" not in message:
            if merged or entry.get("terminal_reason"):
                raise ValueError("Every terminal cluster must list its frame positions.")
            continue
        ids = message["ids"]
        if not isinstance(ids, list) or not ids:
            raise ValueError("Terminal cluster positions must be a nonempty list.")
        if any(isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < sample_num for index in ids):
            raise ValueError("Terminal cluster contains an invalid frame position.")
        if len(set(ids)) != len(ids) or covered[ids].any():
            raise ValueError("Terminal clusters contain overlapping frame positions.")
        covered[ids] = True
        if merged:
            name = entry["cluster_curr"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Merged clusters require a nonempty label.")
            labels[ids] = len(behaviors)
            behaviors.append(name)
    return labels, behaviors


def export_cluster_labels(frame_ids, labels, behaviors, save_path):
    """Write real frame IDs, safely quote names and include unassigned proportions."""
    frame_ids = [str(value).zfill(8) for value in frame_ids]
    if len(frame_ids) != len(labels) or not frame_ids or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Cluster labels need a nonempty, unique frame ID for every row.")
    names = []
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, Integral) or label < UNASSIGNED or label >= len(behaviors):
            raise ValueError("A cluster label is outside the behavior-name range.")
        names.append(UNASSIGNED_NAME if label == UNASSIGNED else behaviors[label])
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("frame_id", "behavior"))
        writer.writerows(zip(frame_ids, names))
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return {name: count / len(names) for name, count in counts.items()}
