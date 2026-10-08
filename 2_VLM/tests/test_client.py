"""Offline tests for the portable inference client; no GPU/server required."""
import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

CLIENT_PATH = Path(__file__).resolve().parents[1] / "prompt_test/client_test.py"
spec = importlib.util.spec_from_file_location("mousegpt_example_client", CLIENT_PATH)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / "imgs").mkdir()
        (self.root / "imgs/mouse.jpg").write_bytes(b"sample image bytes")
        self.jsonl = self.root / "prompts.jsonl"
        self.jsonl.write_text(json.dumps({
            "query": "Describe this mouse.", "images": ["imgs/mouse.jpg"],
            "response": "reference must not be sent",
        }) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_paths_resolve_from_jsonl_and_base64_contains_bytes(self):
        case = client.load_cases(self.jsonl)[0]
        self.assertEqual(case["images"][0], self.root / "imgs/mouse.jpg")
        self.assertEqual(base64.b64decode(client.encode_images(case["images"])[0]), b"sample image bytes")
        self.assertNotIn("response", case)

    def test_missing_image_fails_before_inference(self):
        (self.root / "imgs/mouse.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Line 1: missing image"):
            client.load_cases(self.jsonl)

    def test_bad_query_is_rejected(self):
        self.jsonl.write_text('{"query": null, "images": ["imgs/mouse.jpg"]}\n')
        with self.assertRaisesRegex(ValueError, "query must be a nonempty string"):
            client.load_cases(self.jsonl)

    def test_validate_only_creates_no_outputs(self):
        output = self.root / "results"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(client.main(["--input", str(self.jsonl), "--output-dir", str(output), "--validate-only"]), 0)
        self.assertFalse(output.exists())

    def test_request_and_result_without_server(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Generated description"))])
        mock_api = SimpleNamespace(
            models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="MouseGPT-Lite")])),
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
        output = self.root / "results"
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=lambda **kwargs: mock_api)}):
            with contextlib.redirect_stdout(io.StringIO()):
                client.main(["--input", str(self.jsonl), "--output-dir", str(output)])
        self.assertEqual(calls[0]["messages"], [{"role": "user", "content": "Describe this mouse."}])
        self.assertEqual(calls[0]["model"], "MouseGPT-Lite")
        self.assertEqual(base64.b64decode(calls[0]["extra_body"]["images"][0]), b"sample image bytes")
        self.assertEqual((output / "output_des/0001_mouse.txt").read_text(), "Generated description")


if __name__ == "__main__":
    unittest.main()
