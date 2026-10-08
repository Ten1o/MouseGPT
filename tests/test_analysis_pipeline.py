"""Offline regression tests for Stage 3 parsing, bounded calls, and label alignment."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage

STAGE = Path(__file__).resolve().parents[1] / "3_applications"
sys.path.insert(0, str(STAGE))
packing = importlib.import_module("create_h5")
embedding = importlib.import_module("embedding_request")
fine = importlib.import_module("fine_behaviors")
coarse = importlib.import_module("llm_cluster")
phenotype = importlib.import_module("behavior_phenotype_predict")


def fake_chat(contents):
    values = iter(contents)
    create = Mock(side_effect=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(values)))]))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


def write_description(path, data, wrapped=False):
    if wrapped:
        data = {"choices": [{"message": {"content": "```json\n" + json.dumps(data) + "\n```"}}]}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_json_and_txt_to_embeddings_to_hdf5(tmp_path, monkeypatch):
    folder = tmp_path / "session"
    descriptions = folder / "descriptions"
    descriptions.mkdir(parents=True)
    data = {"Overall": "Mouse is walking.", "Head": "Head faces forward.", "Keywords": "walking"}
    write_description(descriptions / "frame_0001.json", data, wrapped=True)
    write_description(descriptions / "frame_0002.txt", data)
    contents = []

    def provider(content, api_key, **kwargs):
        contents.append(content)
        return np.arange(3072, dtype=float)

    monkeypatch.setattr(embedding, "get_embeddings", provider)
    embedding.process_directory(folder, "fake", max_workers=1)
    assert contents == ["Mouse is walking. Head faces forward."] * 2
    assert sorted(path.name for path in (folder / "feature").glob("*.csv")) == ["frame_0001_des.csv", "frame_0002_des.csv"]
    packing.main(tmp_path, ["session"])
    with pd.HDFStore(tmp_path / "session.h5", mode="r") as store:
        assert store["descriptions"]["id"].tolist() == ["frame_0001", "frame_0002"]
        assert store["descriptions"]["description"].tolist() == [data, data]
        assert list(store["features_des"].index) == ["frame_0001", "frame_0002"]
        np.testing.assert_array_equal(store["features_des"].iloc[0].values, np.arange(3072))


def test_embedding_worker_failure_is_reported(tmp_path, monkeypatch):
    (tmp_path / "descriptions").mkdir()
    write_description(tmp_path / "descriptions/0001.txt", {"Description": "Walking."})
    monkeypatch.setattr(embedding, "get_embeddings", Mock(side_effect=RuntimeError("provider unavailable")))
    with pytest.raises(RuntimeError, match="0001.txt: provider unavailable"):
        embedding.process_directory(tmp_path, "fake")
    assert not list((tmp_path / "feature").iterdir())


def test_embedding_provider_failure_is_bounded(monkeypatch):
    post = Mock(side_effect=embedding.requests.Timeout("timeout"))
    monkeypatch.setattr(embedding.requests, "post", post)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        embedding.get_embeddings("Walking", "fake", max_attempts=2, retry_delay=0)
    assert post.call_count == 2


def test_invalid_embedding_response_is_bounded(monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {"data": [{"embedding": [1, 2, 3]}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(embedding.requests, "post", post)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        embedding.get_embeddings("Walking", "fake", max_attempts=2, retry_delay=0)
    assert post.call_count == 2


def test_incomplete_embeddings_cannot_overwrite_hdf5(tmp_path):
    folder = tmp_path / "session"
    (folder / "descriptions").mkdir(parents=True)
    (folder / "feature").mkdir()
    write_description(folder / "descriptions/one.txt", {"Description": "Walking"})
    write_description(folder / "descriptions/two.txt", {"Description": "Standing"})
    np.savetxt(folder / "feature/one_des.csv", np.zeros(3072))
    original = b"existing output stays unchanged"
    (tmp_path / "session.h5").write_bytes(original)
    with pytest.raises(ValueError, match="missing=.*two"):
        packing.main(tmp_path, ["session"])
    assert (tmp_path / "session.h5").read_bytes() == original


def test_duplicate_description_ids_fail_before_api(tmp_path, monkeypatch):
    (tmp_path / "descriptions").mkdir()
    for suffix in ("json", "txt"):
        write_description(tmp_path / f"descriptions/frame.{suffix}", {"Description": "Walking"})
    provider = Mock()
    monkeypatch.setattr(embedding, "get_embeddings", provider)
    with pytest.raises(ValueError, match="Duplicate description ID"):
        embedding.process_directory(tmp_path, "fake")
    provider.assert_not_called()


@pytest.mark.parametrize("value", ["{}", '{"categories": {"Walking": ["walking"]}}', "not json"])
def test_fine_invalid_responses_are_bounded(monkeypatch, value):
    client, create = fake_chat([value] * 3)
    monkeypatch.setattr(fine, "client", client, raising=False)
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        fine.gpt_request("prompt", max_attempts=3)
    assert create.call_count == 3


def test_invalid_keyword_does_not_create_unbounded_requests(monkeypatch):
    response = json.dumps({"categories": {"Walking": ["walking"]}, "keyword": "running"})
    client, create = fake_chat([response] * 2)
    monkeypatch.setattr(fine, "client", client, raising=False)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        fine.update_categories("prompt", {"Walking": ["walking"]}, "Mouse is standing.", max_attempts=2)
    assert create.call_count == 2


def test_category_update_requires_progress(monkeypatch):
    categories = {"Walking": ["walking"]}
    update = Mock(return_value=categories)
    monkeypatch.setattr(fine, "update_categories", update)
    with pytest.raises(RuntimeError, match="did not reduce"):
        fine.get_labels_category("prompt", categories, ["standing"], "All")
    assert update.call_count == 1


def test_unknown_and_empty_descriptions_do_not_get_first_label():
    labels, counts, unknown = fine.classify_texts(["walking", "standing", "", {}], {"Walking": ["walking"]}, "All")
    assert labels.tolist() == [0, -1, -1, -1]
    assert counts == {"Unknown": 1, "Empty": 2, "Walking": 1}
    assert unknown == ["standing"]


def test_missing_body_part_is_unassigned_instead_of_using_another_part():
    description = {"Torso": "head upward", "Keywords": "head upward"}
    categories = {"Upward": ["head upward"]}
    labels, counts, unknown = fine.classify_texts([description], categories, "Head")
    assert labels.tolist() == [-1]
    assert counts == {"Unknown": 0, "Empty": 1, "Upward": 0}
    assert unknown == []
    all_labels, _, _ = fine.classify_texts([description], categories, "All")
    assert all_labels.tolist() == [0]


def test_distributed_fine_categories_are_valid():
    for file in (STAGE / "init_json").glob("*.json"):
        fine.load_categories(file)


@pytest.mark.parametrize("mapping", [{"0": True}, {"0": "2"}, {"1": "0"}])
def test_coarse_invalid_mappings_are_bounded(monkeypatch, mapping):
    payload = json.dumps({"New Cluster Names": {"0": "Walking"}, "Mapping": mapping})
    request = Mock(return_value=payload)
    monkeypatch.setattr(coarse, "determine_cluster_merges", request)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        coarse.request_cluster_merges({0: ["walking"]}, {}, max_attempts=2)
    assert request.call_count == 2


def test_coarse_can_recover_from_malformed_response(monkeypatch):
    payload = {"New Cluster Names": {"2": "Walking"}, "Mapping": {"4": 2}}
    request = Mock(side_effect=["{}", json.dumps(payload)])
    monkeypatch.setattr(coarse, "determine_cluster_merges", request)
    result = coarse.request_cluster_merges({4: ["walking"]}, {}, max_attempts=2)
    assert result["Mapping"] == {"4": "2"}


def test_coarse_labels_reference_their_category_rows():
    original = np.array([8, -1, 3, 8])
    names = {8: "Walking", -1: "Others", 3: "Standing"}
    categories, labels = coarse.category_tables(original, names, ["a", "b", "c", "d"])
    assert [categories.iloc[index, 0] for index in labels[0]] == ["Walking", "Others", "Standing", "Walking"]
    assert labels.index.tolist() == ["a", "b", "c", "d"]
    assert labels.shape == (4, 1)


def test_impossible_cluster_search_terminates():
    Z = linkage(np.arange(6).reshape(-1, 1))
    with pytest.raises(ValueError, match="cannot fit"):
        coarse.filter_minor_cluster(Z, max_d=3, thresh=3)
    with pytest.raises(ValueError, match="Could not find"):
        coarse.filter_minor_cluster(Z, max_d=2, thresh=3)


def test_coarse_search_does_not_skip_a_valid_intermediate_cut():
    points = np.array([0, 0.1, 0.25, 2, 2.12, 2.28, 10]).reshape(-1, 1)
    labels = coarse.filter_minor_cluster(linkage(points), max_d=2, thresh=3)
    assert labels[-1] == -1
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:6])) == 1
    assert labels[0] != labels[3]
    assert np.unique(labels[labels >= 0], return_counts=True)[1].tolist() == [3, 3]


def test_coarse_rejects_misaligned_rows(tmp_path):
    path = tmp_path / "sample.h5"
    with pd.HDFStore(path, mode="w") as store:
        store.put("descriptions", pd.DataFrame({"id": ["a", "b"], "description": [{"Keywords": "walking"}, {"Keywords": "standing"}]}))
        store.put("features_des", pd.DataFrame(np.ones((2, 3)), index=["b", "a"]))
    with pytest.raises(ValueError, match="not aligned"):
        coarse.load_data([path], tmp_path)


def test_phenotype_context_is_opt_in():
    instructions = phenotype.build_instructions()
    assert "p=0.0252" not in instructions
    assert "User-supplied contextual narrative" not in instructions
    assert "User-supplied contextual narrative" in phenotype.build_instructions("User selected report.")
    groups = phenotype.load_cohorts(STAGE / "report/cohorts.json")
    assert len(groups["knowledge_group"]) == 6
    assert len(groups["test_group"]) == 23


def test_phenotype_failed_response_is_an_error():
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(status="failed")
    with pytest.raises(RuntimeError, match="failed"):
        phenotype.request_prediction(client, "instructions", "prompt", "file")


def test_coarse_pipeline_writes_aligned_hdf5_labels(tmp_path, monkeypatch):
    import openai
    group = "240419_C11M4"
    model = "example"
    folder = tmp_path / model
    folder.mkdir()
    ids = [f"frame_{index:04}" for index in range(8)]
    positions = np.array([[0], [0.1], [0.2], [0.3], [10], [10.1], [10.2], [10.3]])
    h5 = folder / f"{group}.h5"
    with pd.HDFStore(h5, mode="w") as store:
        store.put("descriptions", pd.DataFrame({"id": ids, "description": [{"Keywords": "walking"}] * 8}))
        store.put("features_des", pd.DataFrame(positions, index=ids))
    projection_path, linkage_path = coarse.reduction_paths(tmp_path, ["example"], 1, model, 3, 0.1)
    np.save(projection_path, np.hstack([positions, positions]))
    Z = linkage(positions)
    np.save(linkage_path, Z)
    monkeypatch.setattr(openai, "OpenAI", Mock())
    monkeypatch.setattr(pd.DataFrame, "to_excel", Mock())
    response = {"New Cluster Names": {"7": "Walking", "3": "Standing"}, "Mapping": {"0": "7", "1": "3"}}
    monkeypatch.setattr(coarse, "determine_cluster_merges", Mock(return_value=json.dumps(response)))
    args = SimpleNamespace(downscale=1, max_attempts=2, model_name=model, min_dist=0.1, n_neigh=3, group_config=["example"], workspace=str(tmp_path), npy_folder=str(tmp_path), img_workspace=str(tmp_path), n_clusters=2, min_cluster_size=2, ofolder=str(tmp_path / "output"), date="test")
    coarse.main(args)
    with pd.HDFStore(h5, mode="r") as store:
        labels = store["label_All_cluster_test"]
        categories = store["category_All_cluster_test"]
        assert labels.index.tolist() == ids
        assert labels.shape == (8, 1)
        expected = ["Walking" if label == 0 else "Standing" for label in coarse.filter_minor_cluster(Z, 2, 2)]
        assert [categories.iloc[index, 0] for index in labels[0]] == expected


def test_phenotype_run_retains_instructions_and_holds_out_labels(tmp_path, monkeypatch):
    import openai
    csv = tmp_path / "input.csv"
    csv.write_text("Group,walking_Ratio\ntrain,0.5\ntest,0.2\n")
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"knowledge_group": [["train", "Control"]], "test_group": [["test", "SECRET_TEST_LABEL"]]}))
    context = tmp_path / "context.txt"
    context.write_text("Optional context selected by the user.")
    client = Mock()
    client.files.create.return_value = SimpleNamespace(id="file")
    client.responses.create.return_value = SimpleNamespace(status="completed", id="response", output_text="Prediction: Control")
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=client))
    output = tmp_path / "results"
    phenotype.main(SimpleNamespace(data_csv=str(csv), groups_json=str(groups), context_report=str(context), output=str(output), max_wait=30, model="test-model"))
    arguments = client.responses.create.call_args.kwargs
    assert context.read_text() in arguments["instructions"]
    assert arguments["tools"] == [{"type": "code_interpreter", "container": {"type": "auto", "file_ids": ["file"]}}]
    assert arguments["timeout"] == 30
    assert client.files.create.call_args.kwargs["purpose"] == "user_data"
    prompt = arguments["input"]
    assert "SECRET_TEST_LABEL" not in prompt
    assert "Control" in prompt
    assert "SECRET_TEST_LABEL" in (output / "test.txt").read_text()
    assert json.loads((output / "inputs.json").read_text())["context_sha256"]


def test_phenotype_serializes_responses_code_interpreter_with_real_sdk():
    import httpx2
    import openai

    requests = []

    def handle(request):
        requests.append(request)
        return httpx2.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 1,
            "status": "completed", "model": "gpt-4o",
            "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                        "status": "completed", "content": [
                            {"type": "output_text", "text": "Prediction: Control", "annotations": []}]}],
        })

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
        with openai.OpenAI(api_key="test-only", http_client=transport, max_retries=0) as client:
            response = phenotype.request_prediction(client, "Training-only instructions", "Predict test session",
                                                     "file_test", model="gpt-4o", timeout=17)
    assert response.output_text == "Prediction: Control"
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    payload = json.loads(requests[0].content)
    assert payload["model"] == "gpt-4o"
    assert payload["instructions"] == "Training-only instructions"
    assert payload["tools"] == [{"type": "code_interpreter", "container": {"type": "auto", "file_ids": ["file_test"]}}]
    assert "background" not in payload
    assert requests[0].extensions["timeout"]["read"] == 17


def test_phenotype_sdk_timeout_does_not_retry_prediction():
    import httpx2
    import openai

    calls = []

    def handle(request):
        calls.append(request)
        raise httpx2.ReadTimeout("Simulated timeout", request=request)

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as transport:
        with openai.OpenAI(api_key="test-only", http_client=transport, max_retries=0) as client:
            with pytest.raises(openai.APITimeoutError):
                phenotype.request_prediction(client, "Instructions", "Prompt", "file_test", timeout=1)
    assert len(calls) == 1


def test_embedding_cache_resumes_unchanged_and_regenerates_changed_description(tmp_path, monkeypatch):
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    source = descriptions / "frame_0001.json"
    write_description(source, {"Description": "Walking.", "Keywords": "walking"})
    provider = Mock(side_effect=[np.ones(3072), np.full(3072, 2.0)])
    monkeypatch.setattr(embedding, "get_embeddings", provider)
    output = embedding.process_file(source.name, descriptions, tmp_path / "feature", "secret-key")
    embedding.process_file(source.name, descriptions, tmp_path / "feature", "secret-key")
    assert provider.call_count == 1
    write_description(source, {"Description": "Standing.", "Keywords": "standing"})
    embedding.process_file(source.name, descriptions, tmp_path / "feature", "secret-key")
    assert provider.call_count == 2
    assert provider.call_args.args[0] == "Standing."
    np.testing.assert_array_equal(np.loadtxt(output), np.full(3072, 2.0))
    metadata = (tmp_path / "feature/.embedding-cache/frame_0001.json").read_text()
    assert "secret-key" not in metadata
    assert "Standing." not in metadata
    packing.main(tmp_path.parent, [tmp_path.name])
    with pd.HDFStore(tmp_path.parent / f"{tmp_path.name}.h5", mode="r") as store:
        np.testing.assert_array_equal(store["features_des"].iloc[0].values, np.full(3072, 2.0))


@pytest.mark.parametrize("changed", ["missing_metadata", "invalid_json", "wrong_type", "model", "request_hash", "vector"])
def test_embedding_cache_never_trusts_missing_or_mismatched_provenance(tmp_path, monkeypatch, changed):
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    source = descriptions / "frame.txt"
    write_description(source, {"Description": "Walking."})
    provider = Mock(side_effect=[np.ones(3072), np.full(3072, 3.0)])
    monkeypatch.setattr(embedding, "get_embeddings", provider)
    output = embedding.process_file(source.name, descriptions, tmp_path / "feature", "fake")
    metadata_path = tmp_path / "feature/.embedding-cache/frame.json"
    if changed == "missing_metadata":
        metadata_path.unlink()
    elif changed == "invalid_json":
        metadata_path.write_text("broken")
    elif changed == "wrong_type":
        metadata_path.write_text("[]")
    elif changed in {"model", "request_hash"}:
        metadata = json.loads(metadata_path.read_text())
        metadata["model" if changed == "model" else "request_sha256"] = "different"
        metadata_path.write_text(json.dumps(metadata))
    else:
        np.savetxt(output, np.full(3072, 4.0))
    embedding.process_file(source.name, descriptions, tmp_path / "feature", "fake")
    assert provider.call_count == 2
    np.testing.assert_array_equal(np.loadtxt(output), np.full(3072, 3.0))


def test_interrupted_embedding_metadata_write_cannot_validate_old_cache(tmp_path, monkeypatch):
    descriptions = tmp_path / "descriptions"
    descriptions.mkdir()
    source = descriptions / "frame.txt"
    write_description(source, {"Description": "Walking."})
    provider = Mock(side_effect=[np.ones(3072), np.full(3072, 2.0), np.full(3072, 3.0)])
    monkeypatch.setattr(embedding, "get_embeddings", provider)
    output = embedding.process_file(source.name, descriptions, tmp_path / "feature", "fake")
    write_description(source, {"Description": "Standing."})
    metadata_path = tmp_path / "feature/.embedding-cache/frame.json"
    original_replace = embedding.os.replace

    def interrupt_metadata(source_path, destination):
        if Path(destination) == metadata_path:
            raise OSError("Simulated interrupted metadata write")
        return original_replace(source_path, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(embedding.os, "replace", interrupt_metadata)
        with pytest.raises(OSError, match="interrupted"):
            embedding.process_file(source.name, descriptions, tmp_path / "feature", "fake")
    embedding.process_file(source.name, descriptions, tmp_path / "feature", "fake")
    assert provider.call_count == 3
    np.testing.assert_array_equal(np.loadtxt(output), np.full(3072, 3.0))
