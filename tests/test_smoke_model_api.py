"""The explicit live switch limits model-service smoke tests to one request."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mousegpt_config import ProviderConfig


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("smoke_model_api", ROOT / "scripts/smoke_model_api.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeModelApiTest(unittest.TestCase):
    def config(self, role):
        return ProviderConfig(role, "azure_openai", "https://example.openai.azure.com",
                              "deployment", "2024-12-01-preview", "fake-key")

    def test_dry_run_does_not_import_or_call_requests(self):
        with patch.object(smoke, "load_config", return_value=self.config("AGENT")), patch(
                "requests.post") as post:
            self.assertEqual(smoke.run("AGENT"), 0)
        post.assert_not_called()

    def test_vision_has_one_image_and_small_completion_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "fixture.jpg"
            image.write_bytes(b"offline image fixture")
            operation, body = smoke.request_body(self.config("VISION"), image)
        self.assertEqual(operation, "chat/completions")
        self.assertEqual(body["max_completion_tokens"], 256)
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(len(body["messages"][0]["content"]), 2)

    def test_live_embedding_sends_exactly_one_request(self):
        response = Mock(ok=True, status_code=200)
        response.json.return_value = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        with patch.object(smoke, "load_config", return_value=self.config("EMBEDDING")), patch(
                "requests.post", return_value=response) as post:
            self.assertEqual(smoke.run("EMBEDDING", live=True), 0)
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["input"], "mouse walking")

    def test_compatible_chat_uses_max_tokens(self):
        config = ProviderConfig("AGENT", "anthropic", "https://api.anthropic.com/v1",
                                "claude-model", None, "fake-key")
        operation, body = smoke.request_body(config)
        self.assertEqual(operation, "chat/completions")
        self.assertEqual(body["max_tokens"], 256)
        self.assertNotIn("max_completion_tokens", body)

    def test_official_reasoning_model_uses_completion_budget(self):
        config = ProviderConfig("AGENT", "openai", "https://api.openai.com/v1",
                                "o3", None, "fake-key")
        _, body = smoke.request_body(config)
        self.assertEqual(body["max_completion_tokens"], 256)
        self.assertNotIn("max_tokens", body)

    def test_live_embedding_rejects_zero_or_nonfinite_vectors(self):
        for values in ([0.0, 0.0], [float("nan")], [True], [], ["1.0"]):
            with self.subTest(values=values):
                response = Mock(ok=True, status_code=200)
                response.json.return_value = {"data": [{"embedding": values}]}
                with patch.object(smoke, "load_config", return_value=self.config("EMBEDDING")), patch(
                        "requests.post", return_value=response):
                    self.assertEqual(smoke.run("EMBEDDING", live=True), 1)


if __name__ == "__main__":
    unittest.main()
