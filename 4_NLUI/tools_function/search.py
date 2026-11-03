import numpy as np
import pandas as pd
if __package__:
    from .provider import encode_string_to_feature
    from .similarity import cosine_similarity_matrix, validate_vectors
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from provider import encode_string_to_feature
    from similarity import cosine_similarity_matrix, validate_vectors
from runtime_config import validate_frame_ids
from mousegpt_config import load_config
from mousegpt_embeddings import require_compatible_embeddings


def calculate_cosine_similarity(encoded_feature, vectors):
    similarities = cosine_similarity_matrix(vectors, [encoded_feature])[:, 0]
    order = np.argsort(-similarities, kind="stable")
    return order, similarities[order]


def search(h5_file_path, behavior_description, api_key=None):
    if not isinstance(behavior_description, str) or not behavior_description.strip():
        raise ValueError("The search query must contain text.")
    with pd.HDFStore(h5_file_path, mode="r") as store:
        features = store.get("features_des")
        descriptions = store.get("descriptions")
        frame_ids = validate_frame_ids(descriptions["id"], features.index)
        data = validate_vectors(features.to_numpy(), name="Stored embeddings")
        config = load_config("EMBEDDING", overrides={"api_key": api_key} if api_key else None)
        require_compatible_embeddings(h5_file_path, store, config, data.shape[1])
    encoded_feature = encode_string_to_feature(behavior_description, config=config)
    order, similarities = calculate_cosine_similarity(encoded_feature, data)
    return [frame_ids[index] for index in order], similarities


if __name__ == "__main__":
    print(search("data/240419_C11M4.h5", "walking"))
