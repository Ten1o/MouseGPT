"""Hosted VLM input, provenance, resume and provider-routing regressions."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
import pytest
import requests

from mousegpt_config import ProviderConfig, chat_request_options

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hosted_client", ROOT / "2_VLM/prompt_test/client_test.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def config(model="vision-A", provider="openai_compatible"):
    return ProviderConfig("VISION", provider, "https://vision.example/v1", model, None, "vision-test-key")


def fixture_cases(tmp_path, ids=(8, 16), *, explicit_id=False):
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for frame_id in ids:
        image = source / (f"mouse-{frame_id}.png" if explicit_id else f"{frame_id:08d}.png")
        Image.new("RGB", (16, 12), (100, frame_id % 255, 0)).save(image)
        record = {"query": "Describe visible mouse behavior.", "images": [image.name],
                  "response": "stored-response-must-not-be-sent"}
        if explicit_id:
            record["frame_id"] = frame_id
        records.append(record)
    path = source / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records))
    return path, client.load_cases(path)


def reply():
    values = {field: f"Observed {field}." for field in client.annotation_helpers().DESCRIPTION_FIELDS}
    return {"choices": [{"message": {"content": json.dumps(values)}}]}, values


def successful_post():
    response, _ = reply()
    return Mock(status_code=200, json=lambda: response)


def test_hosted_client_sends_standard_images_and_writes_aligned_outputs(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path)
    post = Mock(return_value=successful_post())
    monkeypatch.setattr(requests, "post", post)
    output = tmp_path / "session"
    assert client.run_api_cases(cases, output, config(), max_attempts=1, timeout=17) == 0
    assert post.call_count == 2
    for call in post.call_args_list:
        assert call.args[0] == "https://vision.example/v1/chat/completions"
        assert call.kwargs["headers"]["Authorization"] == "Bearer vision-test-key"
        body = call.kwargs["json"]
        assert body["model"] == "vision-A"
        assert "images" not in body and "extra_body" not in body
        assert "stored-response-must-not-be-sent" not in json.dumps(body)
        assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert call.kwargs["timeout"] == (10, 17)
    assert sorted(path.stem for path in (output / "descriptions").glob("*.json")) == ["00000008", "00000016"]
    assert sorted(path.stem for path in (output / "images").glob("*.jpg")) == ["00000008", "00000016"]
    with Image.open(output / "images/00000008.jpg") as image:
        assert image.format == "JPEG" and image.size == (16, 12)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["api"]["model"] == "vision-A"
    assert "vision-test-key" not in json.dumps(manifest)


def test_identical_run_resumes_without_provider_and_model_switch_is_rejected(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path)
    post = Mock(return_value=successful_post())
    monkeypatch.setattr(requests, "post", post)
    output = tmp_path / "session"
    client.run_api_cases(cases, output, config())
    client.run_api_cases(cases, output, config())
    assert post.call_count == 2
    with pytest.raises(ValueError, match="new --output-dir"):
        client.run_api_cases(cases, output, config("vision-B"))
    assert post.call_count == 2


def test_modified_results_are_not_overwritten_or_used_as_cache(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path)
    post = Mock(return_value=successful_post())
    monkeypatch.setattr(requests, "post", post)
    output = tmp_path / "session"
    client.run_api_cases(cases, output, config())
    edited = output / "descriptions/00000008.json"
    edited.write_text('{"human": "edited"}')
    with pytest.raises(ValueError, match="were changed"):
        client.run_api_cases(cases, output, config())
    assert post.call_count == 2
    assert edited.read_text() == '{"human": "edited"}'


def test_partial_run_resumes_completed_frames_and_never_retries_401(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path)
    post = Mock(side_effect=[successful_post(), Mock(status_code=401, text="rejected")])
    monkeypatch.setattr(requests, "post", post)
    output = tmp_path / "session"
    with pytest.raises(RuntimeError, match="401"):
        client.run_api_cases(cases, output, config(), max_attempts=3)
    assert post.call_count == 2
    post.side_effect = None
    post.return_value = successful_post()
    client.run_api_cases(cases, output, config())
    assert post.call_count == 3
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"


def test_invalid_response_is_not_published_as_a_description(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path, ids=(8,))
    post = Mock(return_value=Mock(status_code=200, json=lambda: {"choices": [{"message": {"content": "{}"}}]}))
    monkeypatch.setattr(requests, "post", post)
    output = tmp_path / "session"
    with pytest.raises(ValueError, match="Overall"):
        client.run_api_cases(cases, output, config())
    assert post.call_count == 1
    assert not (output / "descriptions/00000008.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"
    failed = list((output / "failed_responses").glob("00000008.*.json"))
    assert len(failed) == 1
    assert json.loads(failed[0].read_text())["choices"][0]["message"]["content"] == "{}"


def test_duplicate_ids_fail_before_output_or_requests(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path, ids=(8,))
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ValueError, match="Duplicate frame ID"):
        client.run_api_cases(cases + cases, tmp_path / "session", config())
    post.assert_not_called()
    assert not (tmp_path / "session").exists()


def test_explicit_frame_id_supports_non_numeric_image_names(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path, ids=(8,), explicit_id=True)
    monkeypatch.setattr(requests, "post", Mock(return_value=successful_post()))
    client.run_api_cases(cases, tmp_path / "session", config())
    assert (tmp_path / "session/descriptions/00000008.json").is_file()


def test_cli_api_path_uses_configured_vision_role(tmp_path, monkeypatch):
    path, _ = fixture_cases(tmp_path)
    roles = []
    def load(role, **kwargs):
        roles.append(role)
        return config()
    monkeypatch.setattr(client, "load_config", load)
    post = Mock(return_value=successful_post())
    monkeypatch.setattr(requests, "post", post)
    assert client.main(["--backend", "api", "--input", str(path), "--limit", "1",
                        "--output-dir", str(tmp_path / "session")]) == 0
    assert roles == ["VISION"]
    assert post.call_count == 1


@pytest.mark.parametrize("model", ["o3", "o4-mini", "gpt-5", "unknown-future-model"])
def test_openai_reasoning_or_unknown_model_omits_sampling_controls(model):
    options = chat_request_options(config(model, "openai"), max_tokens=500,
                                   temperature=0, seed=1, top_p=0.5)
    assert options == {"max_completion_tokens": 500}


def test_known_openai_model_preserves_requested_sampling():
    assert chat_request_options(config("gpt-4o", "openai"), max_tokens=500,
                                temperature=0, seed=1) == {
        "max_completion_tokens": 500, "temperature": 0, "seed": 1}


def test_existing_output_without_manifest_is_preserved(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path)
    output = tmp_path / "session"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data")
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(FileExistsError, match="provenance"):
        client.run_api_cases(cases, output, config())
    post.assert_not_called()
    assert marker.read_text() == "user data"


def test_wrong_role_configuration_is_rejected_before_request(tmp_path, monkeypatch):
    _, cases = fixture_cases(tmp_path)
    incorrect = ProviderConfig("EMBEDDING", "openai_compatible", "https://vectors.example/v1",
                               "vector-model", None, "embedding-only-key")
    post = Mock()
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ValueError, match="VISION configuration"):
        client.run_api_cases(cases, tmp_path / "session", incorrect)
    post.assert_not_called()
