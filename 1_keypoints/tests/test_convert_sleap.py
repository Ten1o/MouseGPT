"""SLEAP format and worker integration checks using synthetic exports."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tables

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('convert_sleap', ROOT/'1_keypoints/demo/convert_sleap.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class SleapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.names = [f'p{i}' for i in range(24)]

    def export(self, matlab=True, count=1, scores=True):
        points = np.arange(3*count*24*2, dtype=float).reshape(3, count, 24, 2)
        points[1, 0, 5] = np.nan
        confidence = np.full((3, count, 24), .75)
        path = self.root/'input.h5'
        with tables.open_file(path, 'w') as out:
            out.create_array('/', 'tracks', points.transpose(1, 3, 2, 0) if matlab else points)
            out.create_array('/', 'node_names', np.array(self.names, dtype='S8'))
            out.create_array('/', 'track_names', np.array([f'mouse{i}' for i in range(count)], dtype='S12'))
            if scores:
                out.create_array('/', 'point_scores', confidence.transpose(1, 2, 0) if matlab else confidence)
        return path, points

    def test_layouts_reordering_missing_and_confidence(self):
        for matlab in (True, False):
            file, points = self.export(matlab)
            records = adapter.read_analysis(file, self.names[::-1])
            self.assertEqual(len(records), 3)
            first = records[0]['instances'][0]
            np.testing.assert_equal(first['keypoints'], points[0, 0, ::-1])
            self.assertEqual(first['keypoint_scores'], [.75]*24)
            missing = records[1]['instances'][0]
            self.assertEqual(missing['keypoints'][18], [0, 0])
            self.assertEqual(missing['keypoint_scores'][18], 0)
            json.dumps(records, allow_nan=False)

    def test_track_selection_and_errors(self):
        file, points = self.export(count=2)
        with self.assertRaisesRegex(ValueError, 'Multiple SLEAP tracks'):
            adapter.read_analysis(file, self.names)
        records = adapter.read_analysis(file, self.names, 'mouse1')
        np.testing.assert_equal(records[0]['instances'][0]['keypoints'], points[0, 1])
        for track in (-1, 2, 'absent'):
            with self.assertRaises(ValueError):
                adapter.read_analysis(file, self.names, track)
        with self.assertRaisesRegex(ValueError, 'missing from node_names'):
            adapter.read_analysis(file, ['absent'], 0)

    def test_annotations_without_scores_and_native_project_error(self):
        file, _ = self.export(scores=False)
        records = adapter.read_analysis(file, self.names)
        self.assertEqual(records[0]['instances'][0]['keypoint_scores'], [1]*24)
        with tables.open_file(self.root/'native.slp', 'w'):
            pass
        with self.assertRaisesRegex(ValueError, 'Export Analysis HDF5'):
            adapter.read_analysis(self.root/'native.slp', self.names)

    def test_invalid_scores_and_layout(self):
        file, _ = self.export()
        with tables.open_file(file, 'a') as out:
            out.root.point_scores[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, 'confidence'):
            adapter.read_analysis(file, self.names)
        with tables.open_file(file, 'a') as out:
            out.remove_node('/tracks')
            out.create_array('/', 'tracks', np.zeros((3, 24, 2)))
        with self.assertRaisesRegex(ValueError, 'four-dimensional'):
            adapter.read_analysis(file, self.names)

    def test_worker_eight_cameras_and_no_overwrite(self):
        sys.path.insert(0, str(ROOT/'ui'))
        import worker
        self.export()
        mapping = self.root/'mapping.json'
        mapping.write_text(json.dumps({'keypoints': self.names,
            'cameras': {f'data{i}': 'input.h5' for i in range(1, 9)}}))
        fields = {'format': 'SLEAP', 'external2d': str(self.root),
                  'mapping': str(mapping), 'out1': str(self.root/'output')}
        result = worker.run('convert', fields, {})
        self.assertEqual(result['frames'], 3)
        self.assertEqual(len(result['files']), 8)
        for file in result['files']:
            self.assertEqual(len(json.loads(Path(file).read_text())), 3)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            worker.run('convert', fields, {})
        # A shorter camera must fail before any camera JSON is written.
        with tables.open_file(self.root/'short.h5', 'w') as out:
            out.create_array('/', 'tracks', np.zeros((1, 2, 24, 1)))
            out.create_array('/', 'node_names', np.array(self.names, dtype='S8'))
        data = json.loads(mapping.read_text())
        data['cameras']['data8'] = 'short.h5'
        mapping.write_text(json.dumps(data))
        fields['out1'] = str(self.root/'mismatched')
        with self.assertRaisesRegex(ValueError, 'frame counts differ'):
            worker.run('convert', fields, {})
        self.assertEqual(list((self.root/'mismatched').glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
