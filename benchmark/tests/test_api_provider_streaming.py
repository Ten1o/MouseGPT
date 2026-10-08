import io
import json
import unittest
from unittest.mock import patch

from mousebench import api_providers as api


def sse(event, *, pretty=False):
    encoded = json.dumps(event, indent=2 if pretty else None, ensure_ascii=False)
    return ("event: " + event["type"] + "\r\n" +
            "\r\n".join("data: " + line for line in encoded.splitlines()) + "\r\n\r\n").encode()


class ResponseStreamTests(unittest.TestCase):
    def test_complete_returns_exact_terminal_response_and_records_events(self):
        result = {"id": "resp-test", "status": "completed", "output_text": "Exact final μouse.\n",
                  "usage": {"input_tokens": 13, "output_tokens": 17}}
        events = [{"type": "response.output_text.delta", "delta": "Different unfinished text"},
                  {"type": "response.completed", "response": result}]
        body = io.BytesIO(b": heartbeat\r\n\r\n" + b"".join(sse(event, pretty=True) for event in events))
        observed = []
        self.assertEqual(api._read_response_body(body, stream=True, on_event=observed.append), result)
        self.assertEqual(observed, events)

    def test_terminal_incomplete_and_failed_receipts_survive_for_persistence(self):
        for status in ("incomplete", "failed"):
            with self.subTest(status=status):
                result = {"status": status, "id": "resp-test", "output": [],
                          "usage": {"output_tokens": 8192}, "incomplete_details": {"reason": "max_output_tokens"}}
                event = {"type": "response." + status, "response": result}
                self.assertEqual(api._read_response_body(io.BytesIO(sse(event)), stream=True), result)
                with self.assertRaises(api.ProviderExecutionError) as raised:
                    api._extract_output(result, "responses")
                self.assertEqual(raised.exception.code, "incomplete_output")

    def test_partial_stream_and_done_marker_never_become_success(self):
        partial = sse({"type": "response.output_text.delta", "delta": "Partial only"})
        for tail in (b"", b"data: [DONE]\n\n", b'data: {"type": "response.completed"}'):
            with self.subTest(tail=tail), self.assertRaises(api.ProviderExecutionError) as raised:
                api._read_response_body(io.BytesIO(partial + tail), stream=True)
            self.assertEqual(raised.exception.code, "incomplete_stream")
            self.assertTrue(raised.exception.uncertain)

    def test_stream_size_is_bounded_even_without_a_newline(self):
        body = io.BytesIO(b"data: " + b"x" * 1000)
        with patch.object(api, "MAX_RESPONSE_BYTES", 128):
            with self.assertRaises(api.ProviderExecutionError) as raised:
                api._read_response_body(body, stream=True)
        self.assertEqual(raised.exception.code, "response_too_large")
        self.assertTrue(raised.exception.uncertain)
        self.assertEqual(body.tell(), 129)

    def test_stream_error_is_recorded_and_safe_error_message_is_bounded(self):
        event = {"type": "error", "message": "secret-echo-never-log"}
        observed = []
        with self.assertRaises(api.ProviderExecutionError) as raised:
            api._read_response_body(io.BytesIO(sse(event)), stream=True, on_event=observed.append)
        self.assertEqual(observed, [event])
        self.assertEqual(raised.exception.code, "provider_stream_error")
        self.assertTrue(raised.exception.uncertain)
        self.assertNotIn("secret-echo", str(raised.exception))

    def test_malformed_terminal_or_json_is_uncertain(self):
        for raw in (b"data: secret-invalid-json\n\n",
                    b'data: {"type": {"malformed": true}}\n\n',
                    sse({"type": "response.completed", "response": {"status": "in_progress"}})):
            with self.subTest(raw=raw), self.assertRaises(api.ProviderExecutionError) as raised:
                api._read_response_body(io.BytesIO(raw), stream=True)
            self.assertTrue(raised.exception.uncertain)
            self.assertNotIn("secret-invalid", str(raised.exception))

    def test_non_streaming_json_is_unchanged_and_bounded(self):
        result = {"status": "completed", "output_text": "Exact output"}
        self.assertEqual(api._read_response_body(io.BytesIO(json.dumps(result).encode())), result)
        with patch.object(api, "MAX_RESPONSE_BYTES", 2), self.assertRaises(api.ProviderExecutionError) as raised:
            api._read_response_body(io.BytesIO(b"{\"oversized\":true}"))
        self.assertEqual(raised.exception.code, "response_too_large")

    def test_stream_setting_is_boolean_responses_only_and_actual_payload_is_logged(self):
        config = {"provider": "openai", "model": "test-model", "settings": {"stream": True, "reasoning_effort": "max"}}
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                api.validate_config(dict(config, settings={"stream": value}))
        with self.assertRaises(ValueError):
            api.validate_config(dict(config, api_mode="chat_completions"))
        messages = [{"role": "user", "content": "Exact frozen prompt.\n"}]
        metadata = [{"attachment_id": "canonical_image", "image_detail": "high", "media_type": "image/jpeg"}]
        for enabled in (True, False):
            selected = api.validate_config(dict(config, settings={"stream": enabled, "reasoning_effort": "max"}))
            with patch.object(api, "_load_job", return_value=(messages, [b"image"], metadata, False)):
                payload, display, _ = api._payload({"task": "inference"}, selected, ".")
            self.assertIs(payload["stream"], enabled)
            self.assertIs(display["stream"], enabled)
            self.assertEqual(payload["input"][0]["content"][0]["text"], messages[0]["content"])
            self.assertEqual(payload["reasoning"], {"effort": "max"})
            self.assertNotIn("text", payload)


if __name__ == "__main__":
    unittest.main()
