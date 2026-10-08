"""Check exported speeds against analytic trajectories in metres and seconds."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "2_VLM"
    / "data_prepareation"
    / "preprocess_imgs.py"
)
spec = importlib.util.spec_from_file_location("preprocess_imgs", MODULE_PATH)
preprocess_imgs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess_imgs)


class PreprocessVelocityTests(unittest.TestCase):
    def export_motion(self, fps, body_velocity, relative_velocities=None):
        """Run preprocessing with real keypoints, bounding boxes, and images."""
        frame_count = 7
        # Include endpoints; image subsampling must not change the keypoint timebase.
        image_frames = (0, 1, 3, 5, 6)
        velocities = np.tile(np.asarray(body_velocity, dtype=float), (24, 1))
        for keypoint, velocity in (relative_velocities or {}).items():
            velocities[keypoint] += velocity
        positions = np.arange(72, dtype=float).reshape(24, 3) / 1000
        times = np.arange(frame_count, dtype=float) / fps
        keypoints = positions[None, :, :] + times[:, None, None] * velocities

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_folder = root / "data"
            json_folder = root / "keypoints"
            json_folder.mkdir()
            np.save(json_folder / "tri_filter.npy", keypoints)
            boxes = [
                {"instances": [{"bbox": [[4, 6, 20, 18]]}]}
                for _ in range(frame_count)
            ]
            for camera in (3, 5):
                image_folder = data_folder / "images" / f"data{camera}"
                image_folder.mkdir(parents=True)
                (json_folder / f"data{camera}.json").write_text(
                    json.dumps(boxes), encoding="utf-8"
                )
                for frame in image_frames:
                    Image.new("RGB", (24, 24), color=(32, 64, 96)).save(
                        image_folder / f"{frame:08d}.jpg"
                    )

            preprocess_imgs.process_data(str(data_folder), str(json_folder), fps)

            motion = json.loads((data_folder / "velocity.json").read_text())
            self.assertEqual(set(motion), {str(frame) for frame in image_frames})
            for camera in (3, 5):
                output = data_folder / "pre_images" / f"data{camera}" / "00000001.jpg"
                with Image.open(output) as cropped:
                    self.assertEqual(cropped.size, (512, 512))
            return motion

    def assert_speeds(self, motion, expected):
        for frame, values in motion.items():
            self.assertEqual(set(values), set(expected))
            for field, speed in expected.items():
                with self.subTest(frame=frame, field=field):
                    self.assertAlmostEqual(values[field], speed, places=9)

    def test_constant_one_metre_per_second_translation_is_100_cm_per_second(self):
        motion = self.export_motion(60.0, (0.6, 0.8, 0.0))
        self.assert_speeds(
            motion,
            {"base": 100.0, "head": 0.0, "forepawL": 0.0,
             "forepawR": 0.0, "hindpawL": 0.0, "hindpawR": 0.0},
        )

    def test_stationary_keypoints_have_zero_speed(self):
        motion = self.export_motion(60.0, (0.0, 0.0, 0.0))
        self.assert_speeds(
            motion,
            {"base": 0.0, "head": 0.0, "forepawL": 0.0,
             "forepawR": 0.0, "hindpawL": 0.0, "hindpawR": 0.0},
        )

    def test_head_and_limbs_report_body_relative_speed(self):
        motion = self.export_motion(
            60.0,
            (0.3, 0.4, 0.0),
            {2: (0.0, 0.0, 0.2), 10: (-0.6, 0.8, 0.0),
             14: (0.0, 0.0, -1.2), 18: (0.03, 0.04, 0.0),
             21: (0.0, -0.25, 0.0)},
        )
        self.assert_speeds(
            motion,
            {"base": 50.0, "head": 20.0, "forepawL": 100.0,
             "forepawR": 120.0, "hindpawL": 5.0, "hindpawR": 25.0},
        )

    def test_two_frames_use_one_sided_difference_at_both_ends(self):
        keypoints = np.zeros((2, 24, 3), dtype=float)
        keypoints[1, :, 0] = 1 / 60
        for frame in (0, 1):
            speed = preprocess_imgs.keypoint_velocity(keypoints, frame, 60)
            np.testing.assert_allclose(speed[:, 0], 100)
            np.testing.assert_allclose(speed[:, 1:], 0)

    def test_speeds_use_supplied_integer_and_fractional_frame_rates(self):
        for fps in (20.0, 59.94):
            with self.subTest(fps=fps):
                motion = self.export_motion(fps, (0.0, 0.0, -1.0))
                self.assert_speeds(
                    motion,
                    {"base": 100.0, "head": 0.0, "forepawL": 0.0,
                     "forepawR": 0.0, "hindpawL": 0.0, "hindpawR": 0.0},
                )


if __name__ == "__main__":
    unittest.main()
