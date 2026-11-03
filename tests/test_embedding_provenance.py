"""Offline model-switching and packaging checks using real session files."""
import importlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3_applications"))
sys.path.insert(0, str(ROOT / "4_NLUI"))
embedding = importlib.import_module("embedding_request")
packing = importlib.import_module("create_h5")
from tools_function import classification, provider, search
import mousegpt_config
import mousegpt_embeddings as provenance


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    monkeypatch.setattr(mousegpt_config, "read_env_file", lambda *args: {})
    monkeypatch.setattr(os, "environ", {
        "MOUSEGPT_API_KEY": "agent-test-key",
        "MOUSEGPT_EMBEDDING_PROVIDER": "openai_compatible",
        "MOUSEGPT_EMBEDDING_ENDPOINT": "https://vectors.example/v1",
        "MOUSEGPT_EMBEDDING_MODEL": "model-A",
        "MOUSEGPT_EMBEDDING_API_KEY": "embedding-test-key",
    })


def make_features(tmp_path, monkeypatch, count=2):
    folder = tmp_path / "session"
    descriptions = folder / "descriptions"
    descriptions.mkdir(parents=True)
    for index, text in enumerate(["resting", "walking"][:count]):
        (descriptions / f"{8 + index * 8:08d}.json").write_text(json.dumps({"Overall": text}))

    def response(url, **kwargs):
        text = kwargs["json"]["input"]
        vector = [1., 0., 0.] if text == "resting" else [0., 1., 0.]
        return Mock(status_code=200, json=lambda: {"data": [{"embedding": vector}]})

    request = Mock(side_effect=response)
    monkeypatch.setattr(embedding.requests, "post", request)
    embedding.process_directory(folder, None, max_workers=1)
    return folder, request


def test_third_party_embeddings_package_query_and_classify(tmp_path, monkeypatch):
    folder, request = make_features(tmp_path, monkeypatch)
    packing.main(tmp_path, [folder.name], with_linkage=True)
    session = tmp_path / "session.h5"
    with pd.HDFStore(session, "r") as store:
        identity = provenance.read_session_embedding_identity(session, store)
        assert identity["model"] == "model-A"
        assert identity["dimensions"] == 3
        assert identity["settings"] == {}
        assert store["linkage_matrix"].shape == (1, 4)
        assert store.get_storer("linkage_matrix").attrs.metric == "cosine"
    result, similarity = search.search(session, "walking")
    assert result == ["00000016", "00000008"]
    np.testing.assert_allclose(similarity, [1., 0.])
    assert classification.classification(session, ["resting", "walking"]) == {
        "00000008": 0, "00000016": 1,
    }
    assert request.call_count == 5
    for call in request.call_args_list:
        assert call.args[0] == "https://vectors.example/v1/embeddings"
        assert call.kwargs["headers"]["Authorization"] == "Bearer embedding-test-key"
    assert "agent-test-key" not in json.dumps(identity)
    assert "embedding-test-key" not in json.dumps(identity)


@pytest.mark.parametrize("operation", ["search", "classification"])
def test_same_dimension_model_switch_is_rejected_before_request(tmp_path, monkeypatch, operation):
    folder, request = make_features(tmp_path, monkeypatch)
    packing.main(tmp_path, [folder.name])
    request.reset_mock()
    monkeypatch.setenv("MOUSEGPT_EMBEDDING_MODEL", "model-B")
    with pytest.raises(ValueError, match="does not match.*model"):
        if operation == "search":
            search.search(tmp_path / "session.h5", "walking")
        else:
            classification.classification(tmp_path / "session.h5", ["walking"])
    request.assert_not_called()


def test_endpoint_switch_is_rejected_before_request(tmp_path, monkeypatch):
    folder, request = make_features(tmp_path, monkeypatch)
    packing.main(tmp_path, [folder.name])
    request.reset_mock()
    monkeypatch.setenv("MOUSEGPT_EMBEDDING_ENDPOINT", "https://other.example/v1")
    with pytest.raises(ValueError, match="does not match.*endpoint"):
        search.search(tmp_path / "session.h5", "walking")
    request.assert_not_called()


def test_single_frame_packaging_has_empty_linkage_table(tmp_path, monkeypatch):
    folder, _ = make_features(tmp_path, monkeypatch, count=1)
    packing.main(tmp_path, [folder.name], with_linkage=True)
    with pd.HDFStore(tmp_path / "session.h5", "r") as store:
        assert store["linkage_matrix"].shape == (0, 4)
        assert store["features_des"].shape == (1, 3)


@pytest.mark.parametrize("change", ["description", "vector", "missing", "identity", "mixed_model"])
def test_packaging_rejects_stale_or_unverified_vectors_without_overwrite(tmp_path, monkeypatch, change):
    folder, _ = make_features(tmp_path, monkeypatch)
    metadata_path = folder / "feature/.embedding-cache/00000008.json"
    if change == "description":
        (folder / "descriptions/00000008.json").write_text(json.dumps({"Overall": "jumping"}))
    elif change == "vector":
        np.savetxt(folder / "feature/00000008_des.csv", [0.5, 0.5, 0.1], delimiter=",")
    elif change == "missing":
        metadata_path.unlink()
    elif change == "identity":
        data = json.loads(metadata_path.read_text())
        data["embedding"]["dimensions"] = 42
        metadata_path.write_text(json.dumps(data))
    else:
        monkeypatch.setenv("MOUSEGPT_EMBEDDING_MODEL", "model-B")
        embedding.process_file("00000008.json", folder / "descriptions", folder / "feature")
    output = tmp_path / "session.h5"
    output.write_bytes(b"preserve existing session")
    with pytest.raises(ValueError, match="provenance|Mixed embedding"):
        packing.main(tmp_path, [folder.name])
    assert output.read_bytes() == b"preserve existing session"


@pytest.mark.parametrize("attribute", [None, "{}", "invalid"])
def test_unknown_legacy_or_invalid_session_metadata_is_rejected(tmp_path, monkeypatch, attribute):
    session = tmp_path / "240419_C11M4.h5"
    with pd.HDFStore(session, "w") as store:
        store.put("features_des", pd.DataFrame([[1., 0., 0.]], index=[8]))
        store.put("descriptions", pd.DataFrame({"id": [8], "description": ["resting"]}))
        if attribute is not None:
            store.get_storer("features_des").attrs.embedding_identity = attribute
    request = Mock(side_effect=AssertionError("No provider call expected"))
    monkeypatch.setattr(provider.requests, "post", request)
    with pytest.raises(ValueError, match="provenance"):
        search.search(session, "walking")
    request.assert_not_called()


def test_legacy_recognition_requires_exact_size_and_hash(tmp_path, monkeypatch):
    path = tmp_path / "legacy.h5"
    path.write_bytes(b"published fixture")
    monkeypatch.setattr(provenance, "LEGACY_DEMO_BYTES", path.stat().st_size)
    monkeypatch.setattr(provenance, "LEGACY_DEMO_SHA256", provenance.file_digest(path))
    provenance._known_legacy_file.cache_clear()
    attrs = Mock(embedding_identity=None)
    store = Mock(get_storer=Mock(return_value=Mock(attrs=attrs)))
    assert provenance.read_session_embedding_identity(path, store)["model"] == "text-embedding-3-large"
    path.write_bytes(b"modified fixture!")
    with pytest.raises(ValueError, match="no verified embedding provenance"):
        provenance.read_session_embedding_identity(path, store)
    provenance._known_legacy_file.cache_clear()


@pytest.mark.parametrize("values", [[0., 0.], [True, 1.], [], [float("nan")], [float("inf")]])
def test_invalid_vectors_are_bounded(values, monkeypatch):
    request = Mock(return_value=Mock(status_code=200, json=lambda: {"data": [{"embedding": values}]}))
    monkeypatch.setattr(embedding.requests, "post", request)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        embedding.get_embeddings("walking", max_attempts=2, retry_delay=0)
    assert request.call_count == 2


def test_consistent_dimensions_are_required_when_packaging(tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    np.savetxt(first, [1., 0.])
    np.savetxt(second, [1., 0., 0.])
    with pytest.raises(ValueError, match="dimensions differ"):
        packing.load_data_matrix([first, second])


def test_corrupt_api_metadata_regenerates_instead_of_sticking_in_cache(tmp_path, monkeypatch):
    folder, request = make_features(tmp_path, monkeypatch, count=1)
    metadata_path = folder / "feature/.embedding-cache/00000008.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["api"]["model"] = "corrupt-model"
    metadata_path.write_text(json.dumps(metadata))
    embedding.process_directory(folder, None, max_workers=1)
    assert request.call_count == 2
    packing.main(tmp_path, [folder.name])


def test_analysis_helper_resolves_analysis_role_without_agent_key(monkeypatch):
    monkeypatch.setenv("MOUSEGPT_ANALYSIS_PROVIDER", "openai_compatible")
    monkeypatch.setenv("MOUSEGPT_ANALYSIS_ENDPOINT", "https://analysis.example/v1")
    monkeypatch.setenv("MOUSEGPT_ANALYSIS_MODEL", "analysis-model")
    monkeypatch.setenv("MOUSEGPT_ANALYSIS_API_KEY", "analysis-test-key")
    request = Mock(return_value=Mock(status_code=200, json=lambda: {"choices": []}))
    monkeypatch.setattr(provider.requests, "post", request)
    provider.post_chat_completion({"messages": []}, validate=lambda value: None)
    assert request.call_args.args[0] == "https://analysis.example/v1/chat/completions"
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer analysis-test-key"
