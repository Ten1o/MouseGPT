"""Validate session descriptions and embeddings, then write aligned HDF5 tables."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mousegpt_embeddings import file_digest, request_digest, validate_identity


from mousegpt_descriptions import parse_description_text, read_description, description_text


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
    dimensions = None
    for file_path in tqdm(csv_paths_list, desc="Loading feature vectors"):
        vector = np.loadtxt(file_path, delimiter=",", ndmin=1)
        if (vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all()
                or not np.any(vector)):
            raise ValueError(f"Expected a finite, nonzero embedding vector in {file_path}; found shape {vector.shape}. Input files were not changed.")
        if dimensions is not None and len(vector) != dimensions:
            raise ValueError(f"Embedding dimensions differ in {file_path}; regenerate embeddings with one model.")
        dimensions = len(vector)
        data.append(vector)
    return np.vstack(data) if data else np.empty((0, 0))


def load_embedding_provenance(folder, ids, descriptions, dimensions, *, feature_folder=None):
    """Verify vector bytes and current descriptions before trusting a sidecar."""
    feature_folder = Path(feature_folder) if feature_folder is not None else Path(folder) / "feature"
    identity = None
    for frame_id, description in zip(ids, descriptions):
        metadata_path = feature_folder / ".embedding-cache" / f"{frame_id}.json"
        vector_path = feature_folder / f"{frame_id}_des.csv"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            current = validate_identity(metadata["embedding"])
            api = {key: current[key] for key in ("provider", "endpoint", "model", "api_version", "settings")}
            valid = (
                current["dimensions"] == dimensions
                and metadata["api"] == api
                and metadata["model"] == current["model"]
                and metadata["vector_sha256"] == file_digest(vector_path)
                and metadata["request_sha256"] == request_digest(api, description_text(description))
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"Missing or invalid embedding provenance for {frame_id}; rerun embedding_request.py before packaging.") from error
        if not valid:
            raise ValueError(f"Stale or mismatched embedding provenance for {frame_id}; rerun embedding_request.py before packaging.")
        if identity is not None and current != identity:
            raise ValueError("Mixed embedding models or settings in this session; regenerate all embeddings with one configuration.")
        identity = current
    return identity


def build_linkage(matrix):
    """Average linkage on cosine distances; quadratic memory for small sessions."""
    if len(matrix) == 1:
        return np.empty((0, 4), dtype=float)
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    # Scaling each nonzero row improves stability without changing cosine
    # distance; rounding error can otherwise produce tiny negative distances.
    scaled = matrix / np.max(np.abs(matrix), axis=1, keepdims=True)
    distances = np.clip(pdist(scaled, metric="cosine"), 0.0, 2.0)
    if not np.isfinite(distances).all():
        raise ValueError("Could not compute finite cosine distances for the session.")
    return linkage(distances, method="average")


def export_session(description_folder, feature_folder, target, *, with_linkage=False,
                   feature_parts_folder=None, overwrite=False):
    """Package independent input paths through one provenance-aware contract.

    A supplied feature directory must contain every vector and its verified
    sidecar. The UI uses the default exclusive output; the historical CLI may
    explicitly replace a session after all inputs have passed validation.
    """
    description_folder, target = Path(description_folder), Path(target)
    if not overwrite and target.exists():
        raise ValueError(f"Output already exists. Choose a new output path: {target}")
    by_id = description_paths_by_id(description_folder)
    ids = sorted(by_id)
    descriptions, _ = load_description_content([by_id[frame_id] for frame_id in ids])
    tables = {"descriptions": pd.DataFrame({"id": ids, "description": descriptions})}
    provenance = None
    if feature_folder is not None:
        feature_folder = Path(feature_folder)
        if not feature_folder.is_dir():
            raise ValueError(f"Embedding directory does not exist: {feature_folder}")
        actual = {path.name[:-len("_des.csv")] for path in feature_folder.glob("*_des.csv")}
        if actual != set(ids):
            missing, extra = sorted(set(ids) - actual), sorted(actual - set(ids))
            raise ValueError(f"Description/embedding IDs do not match: missing={missing}, extra={extra}.")
        matrix = load_data_matrix([feature_folder / f"{frame_id}_des.csv" for frame_id in ids])
        provenance = load_embedding_provenance(
            description_folder.parent, ids, descriptions, matrix.shape[1], feature_folder=feature_folder)
        tables["features_des"] = pd.DataFrame(matrix, index=ids)
    if with_linkage:
        if "features_des" not in tables:
            raise ValueError("--with-linkage requires verified description embeddings in the feature directory.")
        tables["linkage_matrix"] = pd.DataFrame(build_linkage(tables["features_des"].to_numpy()))
    if feature_parts_folder is not None:
        feature_parts_folder = Path(feature_parts_folder)
        for part in ("Head", "Limb", "Torso", "Overall", "Others"):
            matrix = load_data_matrix([feature_parts_folder / f"{frame_id}_des_{part}.csv" for frame_id in ids])
            tables[f"features_{part}"] = pd.DataFrame(matrix, index=ids)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{target.stem}.", suffix=".h5", dir=target.parent)
    os.close(handle)
    try:
        with pd.HDFStore(temporary, mode="w") as store:
            for key, table in tables.items():
                store.put(key, table)
            if provenance is not None:
                store.get_storer("features_des").attrs.embedding_identity = json.dumps(provenance, sort_keys=True)
            if with_linkage:
                store.get_storer("linkage_matrix").attrs.method = "average"
                store.get_storer("linkage_matrix").attrs.metric = "cosine"
        if overwrite:
            os.replace(temporary, target)
        else:
            # Both paths are on the same filesystem. Linking the completed
            # file publishes it atomically and refuses a concurrently-created
            # destination rather than replacing another user's result.
            os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"files": [str(target)], "frames": len(ids),
            "dimensions": provenance["dimensions"] if provenance is not None else 0,
            "embedding": provenance, "with_linkage": bool(with_linkage)}


def main(workspace_folder, groups, *, with_linkage=False):
    workspace = Path(workspace_folder)
    for group in groups:
        folder = workspace / group
        features, parts = folder / "feature", folder / "feature_part"
        target = workspace / f"{group}.h5"
        export_session(folder / "descriptions", features if features.exists() else None,
                       target, with_linkage=with_linkage,
                       feature_parts_folder=parts if parts.exists() else None, overwrite=True)
        print(f"Data saved to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--group_list", nargs="+", required=True)
    parser.add_argument("--with-linkage", action="store_true",
                        help="Create average/cosine hierarchy for Stage 4 clustering (quadratic memory; use small sessions)")
    arguments = parser.parse_args()
    main(arguments.workspace, arguments.group_list, with_linkage=arguments.with_linkage)
