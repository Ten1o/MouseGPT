"""Validate embedding inputs before computing cosine similarities."""
import numpy as np


def validate_vectors(values, *, name="embeddings", dimensions=None):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric matrix.") from error
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a nonempty matrix.")
    if dimensions is not None and array.shape[1] != dimensions:
        raise ValueError(f"{name} has {array.shape[1]} dimensions; expected {dimensions}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values.")
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError(f"{name} must contain finite, nonzero vectors.")
    return array


def cosine_similarity_matrix(data, queries):
    data = validate_vectors(data, name="Stored embeddings")
    queries = validate_vectors(queries, name="Query embeddings", dimensions=data.shape[1])
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return np.clip(data @ queries.T, -1.0, 1.0)
