"""Offline checks for explicit session selection and analysis artifact naming."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("analysis_config", Path(__file__).resolve().parents[1] / "config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


class GroupConfigurationTest(unittest.TestCase):
    def test_example_is_only_the_distributed_session(self):
        self.assertEqual(config.resolve_groups(["example"]), ["240419_C11M4"])

    def test_user_groups_are_not_silently_truncated(self):
        config.group_configs["test_cohort"] = ["session_b", "session_a", "session_b"]
        try:
            self.assertEqual(config.resolve_groups(["test_cohort"]), ["session_a", "session_b"])
        finally:
            del config.group_configs["test_cohort"]

    def test_unknown_or_empty_groups_fail(self):
        for names in ([], ["not_a_published_cohort"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                config.resolve_groups(names)

    def test_artifact_names_include_analysis_settings(self):
        paths = config.reduction_paths("outputs", ["example"], 1, "model", 100)
        self.assertEqual(paths[0].name, "embedding_umap_des_example_ds1_model_100.npy")
        self.assertEqual(paths[1].name, "linkage_matrix_des_example_ds1_model_100.npy")
        other = config.reduction_paths("outputs", ["example"], 1, "model", 100, 0.2)
        self.assertNotEqual(paths, other)
        with self.assertRaises(ValueError):
            config.reduction_paths("outputs", ["example"], 2, "model", 100)


if __name__ == "__main__":
    unittest.main()
