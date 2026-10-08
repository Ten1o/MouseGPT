import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_outliers(data_matrix):
    print("Detecting Outliner for matrix:", data_matrix.shape)
    clf = IsolationForest(random_state=42, n_jobs=-1)
    clf.fit(data_matrix)
    scores = clf.decision_function(data_matrix) # a lower score means more abnormal
    outlier_indices = np.argsort(scores)
    scores = np.sort(scores)
    # is_outlier = clf.predict(data_matrix)
    # outlier_indices = np.where(is_outlier == -1)[0]
    return outlier_indices, scores

def main(h5_file_path):
    with pd.HDFStore(h5_file_path) as store:
        data = np.array(store.get('features_des'))
        outlier_indices, scores = detect_outliers(data)
    return_dict = {}
    for idx, score in zip(outlier_indices, scores):
        return_dict[idx] = score
    return return_dict

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Rank frame rows by IsolationForest anomaly score; lower is more unusual.")
    parser.add_argument("--h5", required=True, help="Prepared HDF5 file with features_des.")
    args = parser.parse_args()
    for row_index, score in main(args.h5).items():
        print(f"{row_index},{score}")