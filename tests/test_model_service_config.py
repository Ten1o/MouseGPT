"""Offline checks for cross-stage provider configuration."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from mousegpt_config import PROVIDER_ENDPOINTS, http_request_config, load_config, read_env_file


class ProviderConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.env_file = Path("unused-test-config")
        self.file_values = {}
        mocked_reader = patch("mousegpt_config.read_env_file",
                              side_effect=lambda path: self.file_values)
        mocked_reader.start()
        self.addCleanup(mocked_reader.stop)

    def test_official_defaults_and_no_environment_mutation(self):
        self.file_values = {"MOUSEGPT_PROVIDER": "openai",
                            "MOUSEGPT_API_KEY": "file-secret"}
        environment = {"UNRELATED": "keep"}
        before = dict(os.environ)
        result = load_config("vision", env=environment, env_file=self.env_file)
        self.assertEqual(result.endpoint, "https://api.openai.com/v1")
        self.assertEqual(result.model, "gpt-4o-2024-05-13")
        self.assertEqual(result.api_key, "file-secret")
        self.assertNotIn("file-secret", repr(result))
        self.assertNotIn("file-secret", str(result.public_summary()))
        self.assertEqual(environment, {"UNRELATED": "keep"})
        self.assertEqual(dict(os.environ), before)

    def test_cli_role_and_shared_precedence(self):
        self.file_values = {
            "MOUSEGPT_PROVIDER": "openai",
            "MOUSEGPT_API_KEY": "file-shared",
            "MOUSEGPT_AGENT_API_KEY": "file-role",
            "MOUSEGPT_AGENT_MODEL": "file-model",
        }
        environment = {
            "MOUSEGPT_API_KEY": "env-shared",
            "MOUSEGPT_AGENT_API_KEY": "env-role",
            "MOUSEGPT_AGENT_MODEL": "env-model",
        }
        result = load_config("agent", env=environment, env_file=self.env_file)
        self.assertEqual((result.api_key, result.model), ("env-role", "env-model"))
        result = load_config("agent", env=environment, env_file=self.env_file,
                             overrides={"api_key": "cli-key", "model": "cli-model"})
        self.assertEqual((result.api_key, result.model), ("cli-key", "cli-model"))
        environment.pop("MOUSEGPT_AGENT_API_KEY")
        self.assertEqual(load_config("agent", env=environment,
                                     env_file=self.env_file).api_key, "file-role")

    def test_legacy_azure_environment_is_a_fallback(self):
        environment = {
            "AZURE_OPENAI_ENDPOINT": "https://mouse.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "legacy-key",
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "legacy-chat",
            "AZURE_OPENAI_CHAT_API_VERSION": "2024-12-01-preview",
        }
        result = load_config("agent", env=environment, env_file=self.env_file)
        self.assertEqual(result.provider, "azure_openai")
        self.assertEqual(result.endpoint, "https://mouse.openai.azure.com")
        self.assertEqual(result.model, "legacy-chat")
        self.assertEqual(result.api_version, "2024-12-01-preview")
        self.assertEqual(result.api_key, "legacy-key")

    def test_different_service_requires_its_own_key(self):
        self.file_values = {
            "MOUSEGPT_PROVIDER": "openai",
            "MOUSEGPT_API_KEY": "official-key",
            "MOUSEGPT_VISION_PROVIDER": "openai_compatible",
            "MOUSEGPT_VISION_ENDPOINT": "https://other.example/v1",
            "MOUSEGPT_VISION_MODEL": "vision-model",
        }
        with self.assertRaisesRegex(ValueError, "MOUSEGPT_VISION_API_KEY"):
            load_config("vision", env={}, env_file=self.env_file)
        result = load_config("vision", env={"MOUSEGPT_VISION_API_KEY": "separate-key"},
                             env_file=self.env_file)
        self.assertEqual(result.api_key, "separate-key")
        self.assertEqual(result.endpoint, "https://other.example/v1")

    def test_bad_endpoint_and_missing_azure_version_fail_before_requests(self):
        with self.assertRaisesRegex(ValueError, "service base URL"):
            load_config("agent", env={}, env_file=self.env_file,
                        overrides={"provider": "openai_compatible",
                                   "endpoint": "https://other.example/v1/chat/completions",
                                   "api_key": "separate-key", "model": "tool-model"})
        with self.assertRaisesRegex(ValueError, "MOUSEGPT_AGENT_API_VERSION"):
            load_config("agent", env={}, env_file=self.env_file,
                        overrides={"provider": "azure_openai",
                                   "endpoint": "https://mouse.openai.azure.com/",
                                   "api_key": "separate-key", "model": "deployment"})

    def test_provider_presets_route_to_their_own_base_urls(self):
        for provider, endpoint in PROVIDER_ENDPOINTS.items():
            if provider == "openai":
                continue
            with self.subTest(provider=provider):
                self.file_values = {"MOUSEGPT_PROVIDER": provider,
                                    "MOUSEGPT_API_KEY": "provider-key",
                                    "MOUSEGPT_AGENT_MODEL": "provider-model"}
                config = load_config("AGENT", env={}, env_file=self.env_file)
                url, headers = http_request_config(config, "chat/completions")
                self.assertEqual(config.endpoint, endpoint)
                self.assertEqual(url, f"{endpoint}/chat/completions")
                self.assertEqual(headers["Authorization"], "Bearer provider-key")

    def test_role_provider_ignores_shared_url_and_requires_dedicated_key(self):
        self.file_values = {"MOUSEGPT_PROVIDER": "openai",
                            "MOUSEGPT_ENDPOINT": "https://api.openai.com/v1",
                            "MOUSEGPT_API_KEY": "official-key",
                            "MOUSEGPT_AGENT_PROVIDER": "anthropic",
                            "MOUSEGPT_AGENT_MODEL": "claude-model"}
        with self.assertRaisesRegex(ValueError, "MOUSEGPT_AGENT_API_KEY"):
            load_config("AGENT", env={}, env_file=self.env_file)
        self.file_values["MOUSEGPT_AGENT_API_KEY"] = "anthropic-key"
        config = load_config("AGENT", env={}, env_file=self.env_file)
        self.assertEqual(config.endpoint, PROVIDER_ENDPOINTS["anthropic"])
        self.assertEqual(config.api_key, "anthropic-key")

    def test_preset_rejects_stale_or_custom_url(self):
        self.file_values = {"MOUSEGPT_PROVIDER": "anthropic",
                            "MOUSEGPT_ENDPOINT": "https://api.openai.com/v1",
                            "MOUSEGPT_API_KEY": "anthropic-key",
                            "MOUSEGPT_AGENT_MODEL": "claude-model"}
        with self.assertRaisesRegex(ValueError, "requires https://api.anthropic.com/v1"):
            load_config("AGENT", env={}, env_file=self.env_file)

    def test_official_legacy_key_is_not_sent_to_new_provider(self):
        self.file_values = {"MOUSEGPT_PROVIDER": "gemini",
                            "MOUSEGPT_AGENT_MODEL": "gemini-model"}
        with self.assertRaisesRegex(ValueError, "MOUSEGPT_AGENT_API_KEY or MOUSEGPT_API_KEY"):
            load_config("AGENT", env={"OPENAI_API_KEY": "official-key"},
                        env_file=self.env_file)

    def test_anthropic_embedding_rejected_and_phenotype_stays_official(self):
        self.file_values = {"MOUSEGPT_PROVIDER": "anthropic",
                            "MOUSEGPT_API_KEY": "anthropic-key",
                            "MOUSEGPT_EMBEDDING_MODEL": "any-model",
                            "MOUSEGPT_PHENOTYPE_MODEL": "claude-model"}
        with self.assertRaisesRegex(ValueError, "does not provide an embedding"):
            load_config("EMBEDDING", env={}, env_file=self.env_file)
        with self.assertRaisesRegex(ValueError, "official OpenAI Responses API"):
            load_config("PHENOTYPE", env={}, env_file=self.env_file)

    def test_literal_env_parser_rejects_duplicate_fields(self):
        with patch.object(Path, "is_file", return_value=True), patch.object(
                Path, "read_text", return_value="MOUSEGPT_API_KEY='value#1'\n"):
            self.assertEqual(read_env_file(self.env_file)["MOUSEGPT_API_KEY"], "value#1")
        with patch.object(Path, "is_file", return_value=True), patch.object(
                Path, "read_text", return_value="MOUSEGPT_API_KEY=one\nMOUSEGPT_API_KEY=two\n"):
            with self.assertRaisesRegex(ValueError, "Duplicate configuration field"):
                read_env_file(self.env_file)


if __name__ == "__main__":
    unittest.main()
