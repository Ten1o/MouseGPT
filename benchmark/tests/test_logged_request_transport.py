"""Recorded request transport: exact candidate inputs and blinded assistance."""
import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mousebench.api_providers import _payload, validate_config
from mousebench.providers import build_job_preview
from mousebench.recorded_inputs import input_semantics


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class LoggedRequestTransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw_images = [b"\xff\xd8\xfforiginal-front-view", b"\xff\xd8\xfforiginal-side-view"]
        images = []
        for index, raw in enumerate(self.raw_images, 1):
            path = f"view{index}.jpg"
            (self.root / path).write_bytes(raw)
            images.append({"attachment_id": f"input_image_{index}", "path": path,
                           "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw),
                           "media_type": "image/jpeg", "image_detail": "high"})
        self.original_system = "  You are the original observer.\r\nKeep these spaces.  "
        self.original_text = "\nDescribe the mouse.\nIts speed is 2.300 cm/s.  \n"
        self.messages = [{"role": "system", "content": self.original_system},
                         {"role": "user", "content": [
                             {"type": "text", "text": self.original_text},
                             {"type": "image_reference", "attachment_id": "input_image_1"},
                             {"type": "text", "text": "\tSecond synchronized view:\n"},
                             {"type": "image_reference", "attachment_id": "input_image_2"}]}]
        self.contract = {"protocol_version": "mousebench.input.v3.logged", "input_hash": "a" * 64,
                         "prompt": self.original_text, "system_prompt": self.original_system,
                         "image_sha256": "c" * 64, "image_layout": "historical_separate_views",
                         "image_detail": "high", "input_images": images, "request_messages": self.messages,
                         "track": "image_velocity", "velocity_embedded_in_prompt": True,
                         "velocity": {"value": 2.3, "units": "cm/s", "source": "PRIVATE velocity source"},
                         "source_request": {"path": "PRIVATE archive path", "model": "PRIVATE MODEL"}}
        self.entry = {"frame_id": "240419_C11M4_00003972", "input_contract": self.contract,
                      "image": {"path": "concat-review.jpg", "sha256": "c" * 64},
                      "expert_evidence": [{"value": "PRIVATE expert rating"}],
                      "candidates": [{"output": "PRIVATE historical answer"}]}
        self.config = validate_config({"provider": "openai", "model": "test-only", "api_mode": "responses"})

    def tearDown(self):
        self.tmp.cleanup()

    def job(self, task="inference", **kwargs):
        self.contract["input_hash"] = digest(input_semantics(self.contract))
        return build_job_preview(task, [self.entry], **kwargs)["jobs"][0]

    def test_chat_reproduces_exact_original_message_forms_order_and_bytes(self):
        request, display, metadata = _payload(self.job(), dict(self.config, api_mode="chat_completions"), self.root)
        expected = copy.deepcopy(self.messages)
        for index, part_position in enumerate((1, 3)):
            expected[1]["content"][part_position] = {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(self.raw_images[index]).decode(), "detail": "high"}}
        self.assertEqual(request["messages"], expected)
        self.assertNotIn("response_format", request)
        self.assertNotIn("Velocity input", json.dumps(request))
        self.assertNotIn("PRIVATE", json.dumps(request))
        self.assertEqual(len(metadata), 2)
        self.assertNotIn("base64", json.dumps(display))
        self.assertEqual(display["messages"][1]["content"][3]["image_url"]["image_reference"]["sha256"],
                         self.contract["input_images"][1]["sha256"])

    def test_responses_preserves_roles_parts_whitespace_and_embedded_velocity_once(self):
        request, _, _ = _payload(self.job(), self.config, self.root)
        self.assertEqual(request["input"][0]["content"], [{"type": "input_text", "text": self.original_system}])
        parts = request["input"][1]["content"]
        self.assertEqual([part["type"] for part in parts], ["input_text", "input_image", "input_text", "input_image"])
        self.assertEqual(parts[0]["text"], self.original_text)
        self.assertEqual(parts[2]["text"], self.messages[1]["content"][2]["text"])
        self.assertEqual(len(request["input"]), 2)
        self.assertNotIn("text", request)
        for index, position in enumerate((1, 3)):
            self.assertEqual(base64.b64decode(parts[position]["image_url"].split(",", 1)[1]), self.raw_images[index])

    def test_sensory_gt_uses_both_views_and_excludes_source_history(self):
        job = self.job("ground_truth", evidence_mode="sensory_only")
        request, _, _ = _payload(job, self.config, self.root)
        text = json.dumps(request)
        self.assertNotIn("PRIVATE", text)
        canonical = json.loads(job["payload_template"]["messages"][1]["content"])["canonical_input"]
        self.assertEqual(canonical["request_messages"], self.messages)
        self.assertEqual(canonical["velocity"], {"value": 2.3, "units": "cm/s"})
        self.assertNotIn("path", canonical["input_images"][0])
        user = next(message for message in request["input"] if message["role"] == "user")
        self.assertEqual([part["type"] for part in user["content"]],
                         ["input_text", "input_image", "input_image"])
        self.assertEqual([item["path"] for item in job["attachment_manifest"]], ["view1.jpg", "view2.jpg"])
        self.assertNotIn("concat-review.jpg", json.dumps(job))

    def test_manifest_order_hash_reference_order_and_original_bytes_are_checked(self):
        for change in ("manifest_order", "manifest_hash", "declared_order", "reference_order", "bytes", "size", "detail"):
            with self.subTest(change=change):
                job = self.job()
                if change == "manifest_order":
                    job["attachment_manifest"].reverse()
                elif change == "manifest_hash":
                    job["attachment_manifest"][1]["sha256"] = "d" * 64
                elif change == "declared_order":
                    job["payload_template"]["attachments"].reverse()
                    job["payload_sha256"] = digest(job["payload_template"])
                elif change == "reference_order":
                    content = json.loads(job["payload_template"]["messages"][1]["content"])
                    parts = content["canonical_input"]["request_messages"][1]["content"]
                    parts[1], parts[3] = parts[3], parts[1]
                    job["payload_template"]["messages"][1]["content"] = json.dumps(content)
                    job["payload_sha256"] = digest(job["payload_template"])
                elif change == "bytes":
                    (self.root / "view2.jpg").write_bytes(b"\xff\xd8\xffrecompressed-other-bytes")
                elif change == "size":
                    job["attachment_manifest"][1]["size"] += 1
                else:
                    job["attachment_manifest"][1]["image_detail"] = "low"
                with self.assertRaises(ValueError):
                    _payload(job, self.config, self.root)
                (self.root / "view2.jpg").write_bytes(self.raw_images[1])

    def test_unbound_or_missing_image_references_and_unsupported_parts_fail_closed(self):
        for change in ("extra", "missing", "duplicate", "url", "unknown_metadata"):
            with self.subTest(change=change):
                entry = copy.deepcopy(self.entry)
                parts = entry["input_contract"]["request_messages"][1]["content"]
                if change == "extra":
                    parts.append({"type": "image_reference", "attachment_id": "input_image_3"})
                elif change == "missing":
                    parts.pop()
                elif change == "duplicate":
                    parts[3]["attachment_id"] = "input_image_1"
                elif change == "url":
                    parts[1] = {"type": "image_url", "image_url": {"url": "https://unbound.example/image.jpg"}}
                else:
                    entry["input_contract"]["request_messages"][0]["extra"] = "metadata"
                with self.assertRaises(ValueError):
                    build_job_preview("inference", [entry])

    def test_per_image_detail_cannot_be_overridden_by_provider(self):
        self.contract["input_images"][1]["image_detail"] = "low"
        request, _, _ = _payload(self.job(), self.config, self.root)
        self.assertEqual(request["input"][1]["content"][3]["detail"], "low")
        with self.assertRaises(ValueError):
            _payload(self.job(), dict(self.config, settings={"image_detail": "high"}), self.root)

    def test_frozen_semantic_hash_rejects_prompt_changes_even_with_new_payload_hash(self):
        job = self.job()
        content = json.loads(job["payload_template"]["messages"][1]["content"])
        content["canonical_input"]["request_messages"][1]["content"][0]["text"] += "changed"
        job["payload_template"]["messages"][1]["content"] = json.dumps(content)
        job["payload_sha256"] = digest(job["payload_template"])
        with self.assertRaisesRegex(ValueError, "frozen input hash"):
            _payload(job, self.config, self.root)
        self.contract["prompt"] += "changed"
        with self.assertRaisesRegex(ValueError, "frozen input hash"):
            build_job_preview("inference", [self.entry])


if __name__ == "__main__":
    unittest.main()
