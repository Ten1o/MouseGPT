"""Offline checks that stage callers use one role-aware configuration file."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mousegpt_config


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageConfigurationTest(unittest.TestCase):
    def test_official_vision_uses_single_shared_key(self):
        vision = load_file("vision_config_test", "2_VLM/data_prepareation/prompt_request.py")
        values = {"MOUSEGPT_PROVIDER": "openai", "MOUSEGPT_API_KEY": "shared-test-key"}
        with patch.dict(os.environ, {}, clear=True), patch.object(
                mousegpt_config, "read_env_file", return_value=values):
            url, headers, identity = vision.resolve_api(None, None)
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer shared-test-key")
        self.assertEqual(identity["model"], "gpt-4o-2024-05-13")

    def test_azure_embedding_can_have_independent_key_and_deployment(self):
        values = {
            "MOUSEGPT_PROVIDER": "openai", "MOUSEGPT_API_KEY": "shared-test-key",
            "MOUSEGPT_EMBEDDING_PROVIDER": "azure_openai",
            "MOUSEGPT_EMBEDDING_ENDPOINT": "https://example.openai.azure.com/",
            "MOUSEGPT_EMBEDDING_API_KEY": "embedding-test-key",
            "MOUSEGPT_EMBEDDING_MODEL": "embedding-deployment",
            "MOUSEGPT_EMBEDDING_API_VERSION": "2024-12-01-preview",
        }
        with patch.dict(os.environ, {}, clear=True), patch.object(
                mousegpt_config, "read_env_file", return_value=values):
            config = mousegpt_config.load_config("EMBEDDING")
            url, headers = mousegpt_config.http_request_config(config, "embeddings")
        self.assertEqual(url, "https://example.openai.azure.com/openai/deployments/"
                         "embedding-deployment/embeddings?api-version=2024-12-01-preview")
        self.assertEqual(headers["api-key"], "embedding-test-key")
        self.assertNotIn("Authorization", headers)

    def test_compatible_analysis_does_not_receive_shared_key(self):
        values = {
            "MOUSEGPT_API_KEY": "shared-test-key",
            "MOUSEGPT_ANALYSIS_PROVIDER": "openai_compatible",
            "MOUSEGPT_ANALYSIS_ENDPOINT": "https://provider.example/v1",
            "MOUSEGPT_ANALYSIS_MODEL": "provider-chat",
        }
        with patch.dict(os.environ, {}, clear=True), patch.object(
                mousegpt_config, "read_env_file", return_value=values):
            with self.assertRaisesRegex(ValueError, "MOUSEGPT_ANALYSIS_API_KEY"):
                mousegpt_config.load_config("ANALYSIS")


if __name__ == "__main__":
    unittest.main()
