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


def create_similarity_matrix(data_matrix, classes_feature):
    return cosine_similarity_matrix(data_matrix, classes_feature)


def classification(h5_file_path, label, api_key):
    if (not isinstance(label, (list, tuple)) or not label
            or any(not isinstance(value, str) or not value.strip() for value in label)):
        raise ValueError("Supply a nonempty list of text labels.")
    with pd.HDFStore(h5_file_path, mode="r") as store:
        features = store.get("features_des")
        descriptions = store.get("descriptions")
    ids = validate_frame_ids(descriptions["id"], features.index)
    data = validate_vectors(features.to_numpy(), name="Stored embeddings")
    classes_feature = [encode_string_to_feature(value, api_key) for value in label]
    similarities = create_similarity_matrix(data, classes_feature)
    assignments = np.argmax(similarities, axis=1)
    return dict(zip(ids, assignments))


if __name__ == "__main__":
    import os
    print(classification("data/240419_C11M4.h5", ["walk", "stand", "stationary"], os.environ["OPENAI_API_KEY"]))
