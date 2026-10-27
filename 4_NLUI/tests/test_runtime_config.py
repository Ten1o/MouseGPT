"""Offline checks for downloaded session selection and credential configuration."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

base = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_config", base / "runtime_config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


class RuntimeConfigurationTest(unittest.TestCase):
    def test_only_complete_sessions_are_offered(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "complete.h5").touch()
            (folder / "incomplete.h5").touch()
            images = folder / "complete_imgs"
            images.mkdir()
            (images / "00000008.jpg").touch()
            self.assertEqual(config.available_sessions(folder), ["complete"])
            selected = config.session_config("complete", folder)
            self.assertEqual(config.image_files(selected)[0].stem, "00000008")
            with self.assertRaises(FileNotFoundError):
                config.session_config("incomplete", folder)

    def test_frame_ids_reject_misalignment_and_filter_unmatched_preview(self):
        self.assertEqual(config.validate_frame_ids([8, 24], ["00000008", "00000024"]), ["00000008", "00000024"])
        with self.assertRaises(ValueError):
            config.validate_frame_ids([8, 24], [24, 8])
        with tempfile.TemporaryDirectory() as folder:
            for frame in ("00000008", "00000016", "00000024"):
                (Path(folder) / f"{frame}.jpg").touch()
            selected = {"img_folder": folder, "frame_ids": ["00000008", "00000024"]}
            self.assertEqual([p.stem for p in config.image_files(selected)], ["00000008", "00000024"])

    def test_session_selection_cannot_escape_data_directory(self):
        for name in ("../outside", "/absolute", "..", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                config.session_config(name)

    def test_key_comes_from_environment_and_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                config.get_api_key()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
            self.assertEqual(config.get_api_key(), "test-key")

    def test_azure_key_and_deployment_config_are_selected_by_endpoint(self):
        environment = {
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "chat-deployment",
            "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
            "OPENAI_API_KEY": "official-key",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertTrue(config.azure_openai_enabled())
            self.assertEqual(config.get_api_key(), "azure-key")
            self.assertEqual(
                config.azure_openai_config(
                    "AZURE_OPENAI_CHAT_DEPLOYMENT",
                    "AZURE_OPENAI_CHAT_API_VERSION",
                    "2024-12-01-preview",
                ),
                ("https://example.openai.azure.com", "chat-deployment",
                 "2024-12-01-preview"),
            )

    def test_tracing_defaults_off_and_respects_explicit_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            config.configure_tracing()
            self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "false")
            self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")
        with patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"}, clear=True):
            config.configure_tracing()
            self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "true")
            self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")

    def test_unfinished_predictor_never_returns_a_fabricated_result(self):
        spec = importlib.util.spec_from_file_location("predictor", base / "tools_function" / "predictor.py")
        predictor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predictor)
        with self.assertRaises(NotImplementedError):
            predictor.predictor("unused.h5", "unused_imgs", "unused-key")


if __name__ == "__main__":
    unittest.main()
