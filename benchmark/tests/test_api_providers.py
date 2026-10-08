import copy
import hashlib
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from mousebench.api_providers import (
    ProviderExecutionError, _NoRedirects, execute_job, provider_status, validate_config,
)


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class APIProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image = b"\xff\xd8\xfffixture-image-bytes"
        (self.root / "image.jpg").write_bytes(self.image)
        self.image_sha = hashlib.sha256(self.image).hexdigest()
        self.canonical = {"prompt": "Describe this mouse exactly.\nOverall: …", "system_prompt": "A fixed system instruction.",
                          "image_detail": "high", "track": "image_only", "image_sha256": self.image_sha}
        self.config = {"provider": "openai", "model": "test-model", "api_key_env": "MOUSEBENCH_TEST_KEY"}
        self.env = patch.dict(os.environ, {"MOUSEBENCH_TEST_KEY": "mock-secret-not-a-real-key"})
        self.env.start()
        self.calls = []

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def job(self, task="inference", extra=None):
        data = {"canonical_input": copy.deepcopy(self.canonical), "archived_secret_label": "NEVER SEND TO INFERENCE"}
        data.update(extra or {})
        payload = {"messages": [{"role": "system", "content": "Generic wrapper NEVER SEND TO INFERENCE"},
                                {"role": "user", "content": json.dumps(data)}],
                   "attachments": [{"attachment_id": "canonical_image", "sha256": self.image_sha}]}
        return {"task": task, "frame_id": "240420_C10M1_00003972", "payload_template": payload,
                "payload_sha256": sha(payload), "input_contract_sha256": "a" * 64,
                "attachment_manifest": [{"attachment_id": "canonical_image", "path": "image.jpg", "sha256": self.image_sha}]}

    def transport(self, url, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        return 200, {"id": "resp_test", "model": "test-model-snapshot", "status": "completed", "created_at": 42,
                     "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"answer":"mouse"}'}]}],
                     "usage": {"input_tokens": 5, "output_tokens": 6}}, {"x-request-id": "req_test"}

    def test_config_status_is_offline_and_contains_no_secret(self):
        with patch("mousebench.api_providers._http_transport") as network:
            status = provider_status(self.config)
        network.assert_not_called()
        self.assertTrue(status["credential_configured"])
        self.assertFalse(status["connection_verified"])
        self.assertNotIn("mock-secret", json.dumps(status))
        self.assertEqual(status["config"]["base_url"], "https://api.openai.com/v1")

    def test_config_rejects_plaintext_credentials_unknown_options_and_unsafe_urls(self):
        for config in (dict(self.config, api_key="secret"), dict(self.config, headers={}),
                       dict(self.config, base_url="http://api.openai.com/v1"),
                       dict(self.config, base_url="https://user:secret@example.com/v1"),
                       dict(self.config, base_url="https://example.com/v1?key=secret"),
                       dict(self.config, base_url="https://127.0.0.1/v1"),
                       dict(self.config, base_url="https://foo.local/v1"),
                       dict(self.config, base_url="https://example.com/../v1"),
                       dict(self.config, base_url="https://example.com/%2e%2e/v1"),
                       dict(self.config, base_url="https://example.com:8443/v1")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_config(config)

    def test_settings_validation_preserves_explicit_options(self):
        settings = {"max_output_tokens": 2000, "temperature": 0, "reasoning_effort": "high", "image_detail": "original", "timeout": 120}
        self.assertEqual(validate_config(dict(self.config, settings=settings))["settings"], settings)
        for key, value in (("temperature", float("nan")), ("max_output_tokens", True),
                           ("reasoning_effort", "surprise"), ("timeout", 0), ("temperature", 3), ("unexpected", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(dict(self.config, settings={key: value}))

    def test_inference_uses_exact_prompt_and_system_and_original_image_only(self):
        result = execute_job(self.job(), self.config, self.root, self.transport)
        payload = self.calls[0][2]
        self.assertEqual(payload["input"][0], {"role": "system", "content": [{"type": "input_text", "text": self.canonical["system_prompt"]}]})
        user = payload["input"][1]
        self.assertEqual(user["content"][0]["text"], self.canonical["prompt"])
        self.assertTrue(user["content"][1]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("NEVER SEND", json.dumps(payload))
        self.assertNotIn("text", payload)
        self.assertFalse(payload["store"])
        self.assertEqual(result["parsed_output"], {"answer": "mouse"})
        log = result["request_log"]
        self.assertEqual(log["kind"], "native_execution")
        self.assertEqual(log["input_sha256"], "a" * 64)
        self.assertEqual(log["observed_model"], "test-model-snapshot")
        self.assertEqual(log["request_id"], "req_test")
        self.assertNotIn("base64", json.dumps(log))
        self.assertNotIn("mock-secret", json.dumps(result))
        self.assertEqual(log["image"]["sha256"], self.image_sha)

    def test_azure_v1_uses_deployment_and_api_key_header(self):
        config = dict(self.config, provider="azure", model="my-deployment", base_url="https://resource.openai.azure.com")
        execute_job(self.job(), config, self.root, self.transport)
        url, headers, payload, timeout = self.calls[0]
        self.assertEqual(url, "https://resource.openai.azure.com/openai/v1/responses")
        self.assertIn("api-key", headers)
        self.assertNotIn("Authorization", headers)
        self.assertEqual(payload["model"], "my-deployment")
        with self.assertRaises(ValueError):
            validate_config(dict(config, base_url="https://wrong.example.com/openai/v1"))

    def test_chat_completions_compatible_mode(self):
        def chat(url, headers, payload, timeout):
            self.calls.append((url, headers, payload, timeout))
            return {"model": "snapshot", "choices": [{"finish_reason": "stop", "message": {"content": "Observed mouse."}}]}
        config = dict(self.config, api_mode="chat_completions", base_url="https://compatible.example.com/v1",
                      settings={"max_output_tokens": 1200, "reasoning_effort": "high"})
        result = execute_job(self.job(), config, self.root, chat)
        url, _, payload, _ = self.calls[0]
        self.assertEqual(url, "https://compatible.example.com/v1/chat/completions")
        self.assertEqual(payload["max_completion_tokens"], 1200)
        self.assertEqual(payload["messages"][1]["content"][1]["type"], "image_url")
        self.assertEqual(result["output_text"], "Observed mouse.")
        self.assertIsNone(result["parsed_output"])

    def test_assistance_and_judging_preserve_template_and_request_json(self):
        for task in ("ground_truth", "gt_check", "judge"):
            with self.subTest(task=task):
                execute_job(self.job(task), self.config, self.root, self.transport)
                payload = self.calls[-1][2]
                self.assertEqual(payload["text"], {"format": {"type": "json_object"}})
                self.assertIn("archived_secret_label", json.dumps(payload))
                self.assertIn("JSON", payload["input"][0]["content"][0]["text"])

    def test_velocity_is_only_permitted_frozen_sensory_information(self):
        self.canonical.update(track="image_velocity", velocity={"value": 2.3, "units": "cm/s", "source": "private-source-path"})
        execute_job(self.job(), self.config, self.root, self.transport)
        payload = self.calls[-1][2]
        self.assertIn("2.3", payload["input"][2]["content"][0]["text"])
        self.assertNotIn("private-source-path", json.dumps(payload))
        self.canonical["track"] = "image_only"
        with self.assertRaises(ValueError):
            execute_job(self.job(), self.config, self.root, self.transport)

    def test_frozen_image_detail_cannot_be_silently_overridden(self):
        self.canonical["image_detail"] = "original"
        execute_job(self.job(), self.config, self.root, self.transport)
        self.assertEqual(self.calls[-1][2]["input"][1]["content"][1]["detail"], "original")
        with self.assertRaises(ValueError):
            execute_job(self.job(), dict(self.config, settings={"image_detail": "high"}), self.root, self.transport)

    def test_payload_hash_image_hash_and_directory_escape_rejected_before_transport(self):
        for change in ("payload", "image_hash", "path", "canonical_hash"):
            job = self.job()
            if change == "payload":
                job["payload_template"]["messages"][0]["content"] = "tampered"
            elif change == "image_hash":
                job["attachment_manifest"][0]["sha256"] = "f" * 64
            elif change == "path":
                job["attachment_manifest"][0]["path"] = "../image.jpg"
            else:
                self.canonical["image_sha256"] = "f" * 64
                job = self.job()
            with self.subTest(change=change), self.assertRaises(ValueError):
                execute_job(job, self.config, self.root, self.transport)
        self.assertEqual(self.calls, [])

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external) / "image.jpg"
            outside.write_bytes(self.image)
            (self.root / "linked.jpg").symlink_to(outside)
            job = self.job()
            job["attachment_manifest"][0]["path"] = "linked.jpg"
            with self.assertRaises(ValueError):
                execute_job(job, self.config, self.root, self.transport)

    def test_missing_credential_makes_no_request(self):
        with patch.dict(os.environ, {"MOUSEBENCH_TEST_KEY": ""}):
            with self.assertRaises(ProviderExecutionError) as exc:
                execute_job(self.job(), self.config, self.root, self.transport)
        self.assertEqual(exc.exception.code, "missing_credential")
        self.assertEqual(self.calls, [])

    def test_timeout_is_uncertain_and_not_retried_or_leaked(self):
        calls = []
        def timeout(*args):
            calls.append(True)
            raise TimeoutError("mock-secret-not-a-real-key")
        with self.assertRaises(ProviderExecutionError) as exc:
            execute_job(self.job(), self.config, self.root, timeout)
        self.assertEqual(len(calls), 1)
        self.assertTrue(exc.exception.uncertain)
        self.assertFalse(exc.exception.retryable)
        self.assertNotIn("mock-secret", json.dumps(exc.exception.as_dict()))

    def test_http_classification_and_no_provider_error_body(self):
        for status, code, uncertain, retryable in ((429, "rate_limited", False, True),
                (503, "provider_failure", True, False), (408, "provider_failure", True, False),
                (401, "request_rejected", False, False), (302, "redirect_blocked", False, False)):
            with self.subTest(status=status), self.assertRaises(ProviderExecutionError) as exc:
                execute_job(self.job(), self.config, self.root,
                            lambda *args: (status, {"error": "mock-secret-not-a-real-key"}, {}))
            self.assertEqual(exc.exception.code, code)
            self.assertEqual(exc.exception.uncertain, uncertain)
            self.assertEqual(exc.exception.retryable, retryable)
            self.assertNotIn("mock-secret", str(exc.exception))

    def test_dns_is_retryable_and_other_connection_failures_uncertain(self):
        for reason, uncertain, retryable in ((socket.gaierror("secret"), False, True), (OSError("secret"), True, False)):
            def fail(*args):
                raise URLError(reason)
            with self.assertRaises(ProviderExecutionError) as exc:
                execute_job(self.job(), self.config, self.root, fail)
            self.assertEqual(exc.exception.uncertain, uncertain)
            self.assertEqual(exc.exception.retryable, retryable)

    def test_private_dns_and_redirect_protection(self):
        with patch("mousebench.api_providers.socket.getaddrinfo", return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]), \
                patch("mousebench.api_providers.build_opener") as opener:
            with self.assertRaises(ProviderExecutionError) as exc:
                execute_job(self.job(), self.config, self.root)
        self.assertEqual(exc.exception.code, "unsafe_endpoint")
        opener.assert_not_called()
        self.assertIsNone(_NoRedirects().redirect_request(None, None, 302, "", {}, "https://other.example.com"))

    def test_response_echo_of_secret_is_redacted(self):
        def echo(*args):
            return {"status": "completed", "model": "test", "output_text": "mock-secret-not-a-real-key", "authorization": "Bearer secret"}
        result = execute_job(self.job(), self.config, self.root, echo)
        self.assertEqual(result["output_text"], "[REDACTED]")
        self.assertEqual(result["raw_response"]["authorization"], "[REDACTED]")

    def test_incomplete_refused_empty_and_nonfinite_outputs(self):
        for response, code in (({"status": "incomplete", "output_text": "partial"}, "incomplete_output"),
                              ({"output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]}, "model_refusal"),
                              ({"output": []}, "empty_output")):
            with self.subTest(code=code), self.assertRaises(ProviderExecutionError) as exc:
                execute_job(self.job(), self.config, self.root, lambda *args: response)
            self.assertEqual(exc.exception.code, code)
        result = execute_job(self.job(), self.config, self.root, lambda *args: {"output_text": '{"score": NaN}'})
        self.assertIsNone(result["parsed_output"])


if __name__ == "__main__":
    unittest.main()
