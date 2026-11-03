"""A self-contained hosted-VLM → embedding → session → search example.

Only HTTP responses are simulated. The public entry points, on-disk outputs,
provenance checks, hierarchy construction and CSV export execute normally.
"""
import csv
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pandas as pd
from PIL import Image
import pytest
import requests

import mousegpt_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3_applications"))
embedding = importlib.import_module("embedding_request")
packing = importlib.import_module("create_h5")


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = load_module("workflow_vlm_client", "2_VLM/prompt_test/client_test.py")
query = load_module("workflow_query_session", "scripts/query_session.py")


def test_documented_api_workflow_keeps_ids_credentials_and_model_identity(tmp_path, monkeypatch):
    settings = {
        "MOUSEGPT_PROVIDER": "openai", "MOUSEGPT_API_KEY": "unused-agent-key",
        "MOUSEGPT_VISION_PROVIDER": "openai_compatible",
        "MOUSEGPT_VISION_ENDPOINT": "https://vision.example/v1",
        "MOUSEGPT_VISION_MODEL": "vision-model",
        "MOUSEGPT_VISION_API_KEY": "vision-only-key",
        "MOUSEGPT_EMBEDDING_PROVIDER": "openai_compatible",
        "MOUSEGPT_EMBEDDING_ENDPOINT": "https://embedding.example/v1",
        "MOUSEGPT_EMBEDDING_MODEL": "embedding-model",
        "MOUSEGPT_EMBEDDING_API_KEY": "embedding-only-key",
    }
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(mousegpt_config, "read_env_file", lambda *args: settings)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    records = []
    for frame, behavior in [(8, "walking"), (16, "grooming"), (24, "resting")]:
        image = inputs / f"{frame:08d}.jpg"
        Image.new("RGB", (16, 12), (frame, 50, 100)).save(image)
        records.append({"query": f"Describe the mouse {behavior} in this frame.",
                        "images": [image.name], "response": "REFERENCE_ONLY_DO_NOT_SEND"})
    prompts = inputs / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps(record) for record in records))
    counts = {"vision": 0, "embedding": 0}

    def respond(url, **kwargs):
        payload = kwargs["json"]
        assert "REFERENCE_ONLY_DO_NOT_SEND" not in json.dumps(payload)
        if url == "https://vision.example/v1/chat/completions":
            assert kwargs["headers"]["Authorization"] == "Bearer vision-only-key"
            assert payload["model"] == "vision-model"
            counts["vision"] += 1
            text = payload["messages"][1]["content"][0]["text"]
            behavior = next(value for value in ("walking", "grooming", "resting") if value in text)
            description = {field: f"The mouse is {behavior}."
                           for field in client.annotation_helpers().DESCRIPTION_FIELDS}
            result = {"choices": [{"message": {"content": json.dumps(description)}}]}
        elif url == "https://embedding.example/v1/embeddings":
            assert kwargs["headers"]["Authorization"] == "Bearer embedding-only-key"
            assert payload["model"] == "embedding-model"
            counts["embedding"] += 1
            text = payload["input"]
            vector = [1.0, 0.0, 0.0] if "walking" in text else [0.0, 1.0, 0.0] if "grooming" in text else [0.0, 0.0, 1.0]
            result = {"data": [{"embedding": vector}]}
        else:
            raise AssertionError(f"Unexpected service: {url}")
        return Mock(status_code=200, json=lambda: result)

    monkeypatch.setattr(requests, "post", respond)
    workspace = tmp_path / "api-demo"
    session_folder = workspace / "demo"
    cli = ["--backend", "api", "--input", str(prompts), "--limit", "3",
           "--max-attempts", "1", "--output-dir", str(session_folder)]
    assert client.main(cli) == 0
    embedding.process_directory(session_folder, None, max_workers=1, max_attempts=1)
    assert counts == {"vision": 3, "embedding": 3}
    packing.main(workspace, ["demo"], with_linkage=True)
    session = workspace / "demo.h5"
    with pd.HDFStore(session, "r") as store:
        assert store["features_des"].shape == (3, 3)
        assert store["descriptions"]["id"].tolist() == ["00000008", "00000016", "00000024"]
        assert store["features_des"].index.tolist() == store["descriptions"]["id"].tolist()
        assert store["linkage_matrix"].shape == (2, 4)
        provenance = json.loads(store.get_storer("features_des").attrs.embedding_identity)
        assert provenance["model"] == "embedding-model"
        assert provenance["endpoint"] == "https://embedding.example/v1"
        assert provenance["dimensions"] == 3
        assert "key" not in json.dumps(provenance)
    assert sorted(path.stem for path in (session_folder / "images").glob("*.jpg")) == ["00000008", "00000016", "00000024"]

    # Validation is local, and both generators resume the unchanged cached data.
    assert query.main(["--session", str(session), "--validate-only"]) == 0
    assert client.main(cli) == 0
    embedding.process_directory(session_folder, None, max_workers=1, max_attempts=1)
    assert counts == {"vision": 3, "embedding": 3}

    output = workspace / "walking.csv"
    assert query.main(["--session", str(session), "--query", "mouse walking",
                       "--top-k", "2", "--output", str(output)]) == 0
    assert counts == {"vision": 3, "embedding": 4}
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["frame_id"] == "00000008"
    assert float(rows[0]["similarity"]) == pytest.approx(1.0)
    assert len(rows) == 2

    # Reusing a result filename is rejected before another query is billed.
    with pytest.raises(SystemExit):
        query.main(["--session", str(session), "--query", "mouse walking", "--output", str(output)])
    assert counts == {"vision": 3, "embedding": 4}
