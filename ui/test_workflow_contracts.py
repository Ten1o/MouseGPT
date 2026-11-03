"""Exercise UI processing against the shared session and pose data contracts."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))
sys.path.insert(0, str(ROOT))
import worker
import mousegpt_config
from mousegpt_embeddings import read_session_embedding_identity


@pytest.fixture
def model_services(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith(("MOUSEGPT_", "OPENAI_", "AZURE_OPENAI_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(mousegpt_config, "read_env_file", lambda *args: {})
    for role in ("EMBEDDING", "AGENT"):
        for name, value in {
            "PROVIDER": "openai_compatible", "ENDPOINT": "https://models.example/v1",
            "MODEL": "vectors-A" if role == "EMBEDDING" else "chat-A",
            "API_KEY": "offline-fixture-key",
        }.items():
            monkeypatch.setenv(f"MOUSEGPT_{role}_{name}", value)

    def response(url, **kwargs):
        assert url == "https://models.example/v1/embeddings"
        content = kwargs["json"]["input"]
        vector = [0., 1., 0.] if "walking" in content else [1., 0., 0.]
        return Mock(status_code=200, json=lambda: {"data": [{"embedding": vector}]})

    request = Mock(side_effect=response)
    monkeypatch.setattr(requests, "post", request)
    return request


def embedded_inputs(tmp_path):
    descriptions, features = tmp_path / "source", tmp_path / "separate-vectors"
    descriptions.mkdir()
    for frame, behavior in ((0, "resting"), (12, "walking")):
        (descriptions / f"{frame:08d}.json").write_text(json.dumps({
            "Overall": behavior, "Head": "Forward", "Limb": "Paws visible",
            "Torso": "Horizontal", "Others": "Tail extended", "Keywords": behavior,
        }))
    fields = {"descriptions": str(descriptions), "features": str(features), "limit": 0}
    assert worker.run("embed", fields, {})["frames"] == 2
    return fields


@pytest.mark.parametrize("hierarchy", ["yes", "no"])
def test_ui_embed_package_validate_and_retrieve(tmp_path, monkeypatch, model_services, hierarchy):
    fields = embedded_inputs(tmp_path)
    session = tmp_path / "different-results" / "session.h5"
    result = worker.run("package", {**fields, "outH5": str(session), "withLinkage": hierarchy}, {})
    assert result["dimensions"] == 3
    assert result["embedding"]["model"] == "vectors-A"
    with pd.HDFStore(session, "r") as store:
        assert read_session_embedding_identity(session, store)["dimensions"] == 3
        assert ("/linkage_matrix" in store.keys()) == (hierarchy == "yes")
    images = tmp_path / "previews"
    images.mkdir()
    for frame in (0, 12):
        Image.new("RGB", (10, 10)).save(images / f"{frame:08d}.jpg")
    state = worker.run("session", {"h5": str(session), "previews": str(images)}, {})
    assert state["matched_images"] == 2
    create = Mock(return_value=SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="Frame 00000012 shows walking."))]))
    monkeypatch.setattr(mousegpt_config, "sdk_client", lambda *a, **k: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    response = worker.run("chat", {"h5": str(session), "frame": 0, "question": "walking"}, {})
    assert response["retrieved_frames"] == ["00000012", "00000000"]
    supplied = json.loads(create.call_args.kwargs["messages"][1]["content"])
    assert supplied["semantic_matches"][0]["frame_id"] == "00000012"
    assert model_services.call_count == 3
    before = session.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        worker.run("package", {**fields, "outH5": str(session)}, {})
    assert session.read_bytes() == before


@pytest.mark.parametrize("change", ["description", "vector", "missing", "mixed"])
def test_ui_packaging_rejects_stale_or_mixed_vectors(tmp_path, monkeypatch, model_services, change):
    fields = embedded_inputs(tmp_path)
    source, features = Path(fields["descriptions"]), Path(fields["features"])
    if change == "description":
        (source / "00000000.json").write_text(json.dumps({"Overall": "jumping"}))
    elif change == "vector":
        np.savetxt(features / "00000000_des.csv", [0., 0., 1.], delimiter=",")
    elif change == "missing":
        (features / ".embedding-cache/00000000.json").unlink()
    else:
        monkeypatch.setenv("MOUSEGPT_EMBEDDING_MODEL", "vectors-B")
        worker.run("embed", {**fields, "limit": 1}, {})
    target = tmp_path / "session.h5"
    with pytest.raises(ValueError, match="provenance|Mixed embedding"):
        worker.run("package", {**fields, "outH5": str(target)}, {})
    assert not target.exists()


def test_ui_chat_rejects_model_change_before_any_request(tmp_path, monkeypatch, model_services):
    fields = embedded_inputs(tmp_path)
    session = tmp_path / "session.h5"
    worker.run("package", {**fields, "outH5": str(session)}, {})
    model_services.reset_mock()
    agent = Mock(side_effect=AssertionError("Agent must not be contacted"))
    monkeypatch.setattr(mousegpt_config, "sdk_client", agent)
    monkeypatch.setenv("MOUSEGPT_EMBEDDING_MODEL", "vectors-B")
    with pytest.raises(ValueError, match="does not match.*model"):
        worker.run("chat", {"h5": str(session), "frame": 0, "question": "walking"}, {})
    model_services.assert_not_called()
    agent.assert_not_called()


def test_ui_validation_rejects_unknown_provenance(tmp_path):
    session = tmp_path / "legacy.h5"
    with pd.HDFStore(session, "w") as store:
        store.put("descriptions", pd.DataFrame({"id": [0], "description": ["resting"]}))
        store.put("features_des", pd.DataFrame([[1., 0.]], index=[0]))
    with pytest.raises(ValueError, match="provenance"):
        worker.aligned_session(session)


def test_analysis_copy_of_recognized_legacy_session_remains_queryable(tmp_path, monkeypatch, model_services):
    import fine_behaviors
    import mousegpt_embeddings as provenance
    source = tmp_path / "published-example.h5"
    with pd.HDFStore(source, "w") as store:
        store.put("descriptions", pd.DataFrame({
            "id": ["00000000", "00000012"],
            "description": [{"Overall": "resting"}, {"Overall": "walking"}],
        }))
        store.put("features_des", pd.DataFrame([[1., 0., 0.], [0., 1., 0.]],
                                               index=["00000000", "00000012"]))
    original = source.read_bytes()
    monkeypatch.setattr(provenance, "LEGACY_DEMO_BYTES", len(original))
    monkeypatch.setattr(provenance, "LEGACY_DEMO_SHA256", provenance.file_digest(source))
    monkeypatch.setattr(provenance, "LEGACY_DEMO_IDENTITY", provenance.embedding_identity(
        mousegpt_config.load_config("EMBEDDING"), 3))
    provenance._known_legacy_file.cache_clear()
    categories, prompt = tmp_path / "categories.json", tmp_path / "prompt.txt"
    categories.write_text(json.dumps({"resting": ["resting"], "walking": ["walking"]}))
    prompt.write_text("Classify descriptions.")
    monkeypatch.setenv("MOUSEGPT_API_KEY", "analysis-offline-key")
    create = Mock(return_value=SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content="Frame 00000012 shows walking."))]))
    monkeypatch.setattr(mousegpt_config, "sdk_client", lambda *a, **k: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    def classify(_prompt, categories, values, part, **kwargs):
        Path(kwargs["output_path"]).write_text(json.dumps(categories))
        return [0, 1], {"resting": 1, "walking": 1}, categories

    monkeypatch.setattr(fine_behaviors, "get_labels_category", classify)
    result = worker.run("analysis:fine", {
        "h5": str(source), "analysisOutput": str(tmp_path / "analysis"),
        "categories": str(categories), "finePrompt": str(prompt), "part": "Overall",
    }, {})
    target = Path(result["files"][0])
    assert target.read_bytes() != original
    assert source.read_bytes() == original
    with pd.HDFStore(target, "r") as store:
        assert store.get_storer("features_des").attrs.embedding_identity
        assert provenance.read_session_embedding_identity(target, store)["model"] == "vectors-A"
    answer = worker.run("chat", {"h5": str(target), "frame": 0, "question": "walking"}, {})
    assert answer["retrieved_frames"][0] == "00000012"


def test_review_load_rejects_source_changed_during_read(tmp_path, monkeypatch):
    import mousegpt_descriptions
    source = tmp_path / "descriptions"
    source.mkdir()
    description = source / "00000000.json"
    description.write_text(json.dumps({"Overall": "resting"}))
    original_reader = mousegpt_descriptions.read_description_snapshot

    def changed_reader(path, **kwargs):
        description.write_text(json.dumps({"Overall": "jumping"}))
        return original_reader(path, **kwargs)

    monkeypatch.setattr(mousegpt_descriptions, "read_description_snapshot", changed_reader)
    with pytest.raises(ValueError, match="Source description changed"):
        worker.run("review", {"descriptions": str(source), "frame": 0}, {})


def test_description_snapshot_supports_wrapped_and_fenced_sources(tmp_path):
    import hashlib
    from mousegpt_descriptions import read_description, read_description_snapshot
    expected = {"Overall": "resting", "Keywords": "resting"}
    for suffix, content in (
        (".txt", "```json\n" + json.dumps(expected) + "\n```"),
        (".json", json.dumps({"choices": [{"message": {"content": json.dumps(expected)}}]})),
    ):
        source = tmp_path / ("description" + suffix)
        source.write_text(content)
        parsed, fingerprint = read_description_snapshot(source)
        assert parsed == expected
        assert fingerprint == hashlib.sha256(source.read_bytes()).hexdigest()
        assert read_description(source, expected_sha256=fingerprint) == expected
        with pytest.raises(ValueError, match="Source description changed"):
            read_description(source, expected_sha256="outdated")


def mmpose_inputs(tmp_path, frame_ids=(0, 1)):
    names = [f"point-{index}" for index in range(24)]
    source_names = list(reversed(names))
    mapping = {"keypoints": names, "source_keypoints": source_names,
               "cameras": {f"data{i}": f"camera{i}.json" for i in range(1, 9)}}
    source = tmp_path / "mmpose"
    source.mkdir()
    for filename in mapping["cameras"].values():
        records = [{"frame_id": frame, "instances": [{
            "keypoints": [[index, index + 1] for index in range(24)],
            "keypoint_scores": [0.9] * 24,
        }]} for frame in frame_ids]
        (source / filename).write_text(json.dumps(records))
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps(mapping))
    fields = {"external2d": str(source), "mapping": str(mapping_file),
              "out1": str(tmp_path / "converted"), "format": "MMPose JSON"}
    return fields, mapping


def test_mmpose_import_reorders_nodes_and_preserves_original_frames(tmp_path):
    fields, mapping = mmpose_inputs(tmp_path)
    # Per-camera source orders are supported as well as one common order.
    mapping["source_keypoints"] = {name: mapping["source_keypoints"] for name in mapping["cameras"]}
    Path(fields["mapping"]).write_text(json.dumps(mapping))
    result = worker.run("convert", fields, {})
    assert result["frames"] == 2
    converted = json.loads(Path(result["files"][0]).read_text())
    assert [frame["frame_id"] for frame in converted] == [0, 1]
    assert converted[0]["instances"][0]["keypoints"][0] == [23., 24.]
    assert converted[0]["instances"][0]["keypoints"][-1] == [0., 1.]


@pytest.mark.parametrize("frame_ids", [(120, 132), (0, 2), (1, 0), (0, 0), ("0", "1")])
def test_mmpose_import_rejects_offset_sparse_or_reordered_frames(tmp_path, frame_ids):
    fields, _ = mmpose_inputs(tmp_path, frame_ids)
    with pytest.raises(ValueError, match="frame_id"):
        worker.run("convert", fields, {})
    assert not list(Path(fields["out1"]).glob("*.json"))


def test_mmpose_import_requires_source_node_order(tmp_path):
    fields, mapping = mmpose_inputs(tmp_path)
    del mapping["source_keypoints"]
    Path(fields["mapping"]).write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="source_keypoints"):
        worker.run("convert", fields, {})


@pytest.mark.parametrize("existing", ["data1.json", "preview/data8.mp4"])
def test_pose_rejects_existing_outputs_before_starting_any_camera(tmp_path, monkeypatch, existing):
    videos = tmp_path / "videos"
    videos.mkdir()
    destination = tmp_path / "predictions"
    saved = destination / existing
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"previous prediction")
    process = Mock(side_effect=AssertionError("Do not start inference"))
    monkeypatch.setattr(worker.subprocess, "run", process)
    with pytest.raises(ValueError, match="not empty"):
        worker.run("pose", {"videos": str(videos), "out1": str(destination)}, {})
    assert saved.read_bytes() == b"previous prediction"
    process.assert_not_called()
