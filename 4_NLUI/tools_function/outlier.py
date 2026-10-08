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

def outlier(h5_file_path):
    with pd.HDFStore(h5_file_path, mode='r') as store:
        data = np.array(store.get('features_des'))
        outlier_indices_raw, scores = detect_outliers(data)

    out_index = []
    with pd.HDFStore(h5_file_path, mode='r') as store:
        data = store.get('descriptions')
        for index in outlier_indices_raw:
            out_index.append(data['id'].iloc[index])

    return out_index, scores

if __name__ == '__main__':
    h5_file_path = 'data/240419_C11M4.h5'
    print(outlier(h5_file_path))