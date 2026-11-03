"""Convert SLEAP Analysis HDF5 exports to the project's per-frame JSON schema.

Supports SLEAP's MATLAB layout and sleap-io's standard layout. No SLEAP
installation is required. Native .slp projects must first be exported in SLEAP.
"""
from pathlib import Path

import tables
import numpy as np


def read_analysis(filename, keypoints, track=None):
    """Return one selected animal, preserving every exported frame including gaps.

    track is a zero-based index or an exact track name. Multiple exported tracks
    require explicit selection. Missing points become (0, 0) with confidence 0.
    """
    filename = Path(filename)
    with tables.open_file(filename, 'r') as handle:
        source = handle.root._v_children
        if not {'tracks', 'node_names'} <= set(source):
            raise ValueError('Expected SLEAP Analysis HDF5. Export the .slp project using File > Export Analysis HDF5 in SLEAP.')
        def strings(dataset):
            return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in dataset.read().tolist()]
        nodes = strings(source['node_names'])
        if len(set(nodes)) != len(nodes):
            raise ValueError('SLEAP node_names contains duplicates.')
        missing = [name for name in keypoints if name not in nodes]
        if missing:
            raise ValueError(f'SLEAP keypoints missing from node_names: {missing}')
        points = source['tracks'][:]
        if points.ndim != 4:
            raise ValueError('SLEAP tracks must be a four-dimensional array.')
        # The coordinate axis distinguishes the two official layouts because
        # the UI requires 24 mapped nodes. Never guess between arbitrary axes.
        matlab = points.shape[1] == 2 and points.shape[2] == len(nodes)
        standard = points.shape[-1] == 2 and points.shape[2] == len(nodes)
        track_names = strings(source['track_names']) if 'track_names' in source else []
        if track_names:
            matlab = matlab and points.shape[0] == len(track_names)
            standard = standard and points.shape[1] == len(track_names)
        if 'point_scores' in source:
            shape = source['point_scores'].shape
            matlab = matlab and shape == (points.shape[0], points.shape[2], points.shape[3])
            standard = standard and shape == points.shape[:3]
        if matlab == standard:
            raise ValueError('Unsupported or ambiguous SLEAP tracks layout; use a standard Analysis HDF5 export.')
        if matlab:
            points = points.transpose(3, 0, 2, 1)
        frames, count, node_count, _ = points.shape
        if frames == 0 or count == 0:
            raise ValueError('SLEAP export contains no frames or tracks.')
        if track is None:
            if count != 1:
                raise ValueError('Multiple SLEAP tracks: select a zero-based track index or name in mapping.json tracks for this camera.')
            index = 0
        elif isinstance(track, int) and not isinstance(track, bool):
            index = track
        elif isinstance(track, str) and track in track_names:
            index = track_names.index(track)
        else:
            raise ValueError(f'Unknown SLEAP track: {track!r}')
        if not 0 <= index < count:
            raise ValueError(f'SLEAP track index must be between 0 and {count - 1}.')
        if 'point_scores' in source:
            scores = source['point_scores'][:]
            if matlab:
                scores = scores.transpose(2, 0, 1)
            if scores.shape != (frames, count, node_count):
                raise ValueError('SLEAP point_scores shape does not match tracks.')
        else:
            # Manual annotations may have no confidence dataset.
            scores = np.ones((frames, count, node_count))
        order = [nodes.index(name) for name in keypoints]
        xy = points[:, index][:, order]
        confidence = scores[:, index][:, order]
        valid = np.isfinite(xy).all(axis=-1) & np.isfinite(confidence)
        if np.any((confidence[valid] < 0) | (confidence[valid] > 1)):
            raise ValueError('SLEAP point confidence must be between 0 and 1.')
        confidence = np.where(valid, confidence, 0)
        xy = np.where(valid[..., None], xy, 0)
        records = []
        for coordinates, score in zip(xy, confidence):
            visible = coordinates[score > 0]
            bbox = [*visible.min(axis=0).tolist(), *visible.max(axis=0).tolist()] if len(visible) else [0, 0, 0, 0]
            records.append({'instances': [{'keypoints': coordinates.tolist(),
                            'keypoint_scores': score.tolist(), 'bbox': [bbox]}]})
        return records
