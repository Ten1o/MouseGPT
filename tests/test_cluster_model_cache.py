"""Changing analysis configuration must invalidate cached model decisions."""
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import linkage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_NLUI"))
cluster_module = importlib.import_module("tools_function.cluster")
from mousegpt_config import ProviderConfig


@pytest.mark.parametrize("changed", ["model", "endpoint", "provider", "api_version"])
def test_changed_analysis_identity_recomputes_cache(tmp_path, monkeypatch, changed):
    path = tmp_path / "session.h5"
    with pd.HDFStore(path, "w") as store:
        store.put("features_des", pd.DataFrame([[1., 0.], [0., 1.]], index=[8, 16]))
        store.put("descriptions", pd.DataFrame({"id": [8, 16], "description": ["a", "b"]}))
        store.put("linkage_matrix", pd.DataFrame(linkage([[1., 0.], [0., 1.]])))
    identity = dict(role="ANALYSIS", provider="openai", endpoint="https://api.openai.com/v1",
                    model="model-A", api_version=None, api_key="private-test-key")
    monkeypatch.setattr(cluster_module, "load_config", lambda *a, **kw: ProviderConfig(**identity))
    monkeypatch.setattr(cluster_module, "__file__", str(tmp_path / "tools_function/cluster.py"))
    calls = []

    def decision(*args, **kwargs):
        calls.append(dict(identity))
        return {0: {"level": 1, "cluster_curr": f"result-{len(calls)}",
                    "message": {"merge": True, "ids": [0, 1]}}}

    monkeypatch.setattr(cluster_module, "bfs_traverse_print", decision)
    first = cluster_module.cluster(path, tmp_path)
    repeated = cluster_module.cluster(path, tmp_path)
    assert first[1] == repeated[1] == ["result-1"]
    assert len(calls) == 1
    identity[changed] = {"model": "model-B", "endpoint": "https://other.example/v1",
                         "provider": "openai_compatible", "api_version": "version-B"}[changed]
    second = cluster_module.cluster(path, tmp_path)
    assert second[1] == ["result-2"]
    assert len(calls) == 2
    assert all("private-test-key" not in p.read_text() for p in (tmp_path / "workspace/cluster_cache").glob("*.json"))
