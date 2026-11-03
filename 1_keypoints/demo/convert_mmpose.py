"""Convert explicit, frame-aligned single-animal MMPose video predictions.

The mapping supplies the export's node order as ``source_keypoints`` and the
MouseGPT target order as ``keypoints``. Frame IDs must cover original frames
0..N-1; sparse, shifted or reordered exports cannot be reconstructed by row.
"""
import json
from pathlib import Path

import numpy as np


def read_predictions(filename, keypoints, source_keypoints):
    if (not isinstance(source_keypoints, list) or len(source_keypoints) != 24
            or not all(isinstance(name, str) and name for name in source_keypoints)
            or len(set(source_keypoints)) != 24 or set(source_keypoints) != set(keypoints)):
        raise ValueError(
            "MMPose mapping requires source_keypoints: 24 unique names in export order, "
            "matching the target keypoints names. Supply a list or one list per camera.")
    data = json.loads(Path(filename).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("MMPose video predictions must be a nonempty list of frame records.")
    order = [source_keypoints.index(name) for name in keypoints]
    records = []
    for index, frame in enumerate(data):
        if (not isinstance(frame, dict) or type(frame.get("frame_id")) is not int
                or frame["frame_id"] != index):
            raise ValueError(
                "MMPose frame_id must be consecutive original frame IDs starting at 0; "
                "sparse, offset or reordered exports would misalign videos and reconstruction.")
        instances = frame.get("instances")
        if not isinstance(instances, list) or len(instances) != 1:
            raise ValueError("MMPose import requires exactly one selected animal in every original frame.")
        instance = instances[0]
        if not isinstance(instance, dict):
            raise ValueError("Invalid MMPose animal instance.")
        points = np.asarray(instance.get("keypoints"), dtype=float)
        scores = np.asarray(instance.get("keypoint_scores"), dtype=float)
        if points.shape != (24, 2) or scores.shape != (24,):
            raise ValueError("Expected 24 keypoints and confidence scores per MMPose frame.")
        if not np.isfinite(points).all() or not np.isfinite(scores).all():
            raise ValueError("MMPose points and confidence scores must be finite; resolve missing points first.")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("MMPose keypoint confidence must be between 0 and 1.")
        points, scores = points[order], scores[order]
        visible = points[scores > 0]
        bbox = ([*visible.min(axis=0).tolist(), *visible.max(axis=0).tolist()]
                if len(visible) else [0, 0, 0, 0])
        records.append({"frame_id": index, "instances": [{
            "keypoints": points.tolist(), "keypoint_scores": scores.tolist(), "bbox": [bbox],
        }]})
    return records
