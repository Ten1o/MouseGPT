"""Offline checks for OpenAI and Azure OpenAI HTTP request selection."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base))

from tools_function import provider
import mousegpt_config


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


class ProviderConfigurationTest(unittest.TestCase):
    def test_azure_embedding_uses_deployment_url_and_api_key_header(self):
        environment = {
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "embedding-deployment",
            "AZURE_OPENAI_EMBEDDING_API_VERSION": "2023-05-15",
        }
        response = FakeResponse({"data": [{"embedding": [1.0]}]})
        with patch.object(mousegpt_config, "read_env_file", return_value={}), patch.dict(os.environ, environment, clear=True), patch.object(
                provider.requests, "post", return_value=response) as request:
            self.assertEqual(provider.encode_string_to_feature("walking", "secret"), [1.0])
        _, kwargs = request.call_args
        self.assertEqual(
            request.call_args.args[0],
            "https://example.openai.azure.com/openai/deployments/"
            "embedding-deployment/embeddings?api-version=2023-05-15",
        )
        self.assertEqual(kwargs["headers"]["api-key"], "secret")
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertEqual(kwargs["json"], {"input": "walking"})

    def test_azure_chat_removes_official_model_and_sampling_fields(self):
        environment = {
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_CHAT_DEPLOYMENT": "chat-deployment",
            "AZURE_OPENAI_CHAT_API_VERSION": "2024-12-01-preview",
        }
        response = FakeResponse({"choices": []})
        with patch.object(mousegpt_config, "read_env_file", return_value={}), patch.dict(os.environ, environment, clear=True), patch.object(
                provider.requests, "post", return_value=response) as request:
            provider.post_chat_completion(
                {
                    "model": "gpt-4o",
                    "messages": [],
                    "temperature": 0.3,
                    "top_p": 0.5,
                    "max_tokens": 2000,
                },
                api_key="secret",
                validate=lambda result: None,
            )
        _, kwargs = request.call_args
        self.assertNotIn("model", kwargs["json"])
        self.assertNotIn("temperature", kwargs["json"])
        self.assertNotIn("top_p", kwargs["json"])
        self.assertEqual(kwargs["json"]["max_completion_tokens"], 2000)


if __name__ == "__main__":
    unittest.main()
