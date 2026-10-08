"""Validate session descriptions and embeddings, then write aligned HDF5 tables."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_description_text(content):
    """Read plain text or a JSON object, including a fenced model response."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Description must be nonempty text or a JSON object.")
    content = content.strip()
    if "{" not in content:
        if content.startswith("```") or content.endswith("}"):
            raise ValueError("Malformed JSON description.")
        return content
    start = content.index("{")
    try:
        value, end = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed JSON description.") from exc
    trailing = content[start + end:].strip()
    if trailing not in ("", "```"):
        raise ValueError("Unexpected content after description JSON.")
    if not isinstance(value, dict) or not value:
        raise ValueError("Description JSON must be a nonempty object.")
    return value


def read_description(path):
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".txt":
        value = parse_description_text(content)
    else:
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}") from exc
        if isinstance(value, dict) and "choices" in value:
            choices = value["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"No valid description in {path}")
            value = parse_description_text(choices[0].get("message", {}).get("content"))
        if not isinstance(value, (str, dict)):
            raise ValueError(f"Invalid description in {path}")
    description_text(value)
    return value


def description_text(description):
    """Use the same descriptive fields for JSON and text inputs, excluding keywords."""
    if isinstance(description, dict):
        values = [value for key, value in description.items() if key != "Keywords"]
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Description fields must contain strings.")
        content = " ".join(value.strip() for value in values if value.strip())
    elif isinstance(description, str):
        content = description.strip()
    else:
        raise ValueError("Description must be text or an object.")
    if not content:
        raise ValueError("Description has no text to embed.")
    return content


def load_description_txt(desc_path):
    return read_description(desc_path)


def load_description_content(description_json_list):
    descriptions, ids = [], []
    for path in tqdm(description_json_list, desc="Loading description content"):
        descriptions.append(read_description(path))
        ids.append(Path(path).stem)
    return descriptions, ids


def description_paths_by_id(folder):
    paths = sorted(path for path in Path(folder).iterdir() if path.suffix.lower() in {".txt", ".json"} and path.is_file())
    if not paths:
        raise FileNotFoundError(f"No descriptions found in {folder}")
    by_id = {}
    for path in paths:
        if path.stem in by_id:
            raise ValueError(f"Duplicate description ID: {path.stem}")
        by_id[path.stem] = path
    return by_id


def load_data_matrix(csv_paths_list):
    data = []
    for file_path in tqdm(csv_paths_list, desc="Loading feature vectors"):
        vector = np.loadtxt(file_path, delimiter=",")
        if vector.shape != (3072,) or not np.isfinite(vector).all():
            raise ValueError(f"Expected 3072 finite embedding values in {file_path}; found shape {vector.shape}. Input files were not changed.")
        data.append(vector)
    return np.vstack(data) if data else np.empty((0, 3072))


def main(workspace_folder, groups):
    workspace = Path(workspace_folder)
    for group in groups:
        folder = workspace / group
        by_id = description_paths_by_id(folder / "descriptions")
        ids = sorted(by_id)
        descriptions, _ = load_description_content([by_id[frame_id] for frame_id in ids])
        tables = {"descriptions": pd.DataFrame({"id": ids, "description": descriptions})}
        feature_folder = folder / "feature"
        if feature_folder.exists():
            actual = {path.name[:-len("_des.csv")] for path in feature_folder.glob("*_des.csv")}
            if actual != set(ids):
                missing, extra = sorted(set(ids) - actual), sorted(actual - set(ids))
                raise ValueError(f"Description/embedding IDs do not match: missing={missing}, extra={extra}.")
            matrix = load_data_matrix([feature_folder / f"{frame_id}_des.csv" for frame_id in ids])
            tables["features_des"] = pd.DataFrame(matrix, index=ids)
        if (folder / "feature_part").exists():
            for part in ("Head", "Limb", "Torso", "Overall", "Others"):
                matrix = load_data_matrix([folder / "feature_part" / f"{frame_id}_des_{part}.csv" for frame_id in ids])
                tables[f"features_{part}"] = pd.DataFrame(matrix, index=ids)
        target = workspace / f"{group}.h5"
        handle, temporary = tempfile.mkstemp(prefix=f".{group}.", suffix=".h5", dir=workspace)
        os.close(handle)
        try:
            with pd.HDFStore(temporary, mode="w") as store:
                for key, table in tables.items():
                    store.put(key, table)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"Data saved to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--group_list", nargs="+", required=True)
    arguments = parser.parse_args()
    main(arguments.workspace, arguments.group_list)
