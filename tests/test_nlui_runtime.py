"""Exercise the installed interface and its real agent/tool wiring offline."""
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pandas as pd
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "4_NLUI"))


@pytest.fixture
def interface(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    pytest.importorskip("gradio")
    pytest.importorskip("langchain_classic")
    pytest.importorskip("langchain_openai")
    import mousegpt_config
    import runtime_config
    from mousegpt_embeddings import embedding_identity

    settings = {"MOUSEGPT_API_KEY": "embedding-test-key",
                "MOUSEGPT_AGENT_PROVIDER": "openai_compatible",
                "MOUSEGPT_AGENT_ENDPOINT": "https://agent.example/v1",
                "MOUSEGPT_AGENT_API_KEY": "agent-test-key",
                "MOUSEGPT_AGENT_MODEL": "tool-capable-model"}
    monkeypatch.setattr(mousegpt_config, "read_env_file", lambda *a: settings)
    # No shell credential can influence this offline test.
    for key in list(__import__("os").environ):
        if key.startswith(("MOUSEGPT_", "OPENAI_", "AZURE_OPENAI_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(runtime_config, "WORKSPACE_DIR", tmp_path / "workspace")
    module = importlib.import_module("agent")
    data = tmp_path / "data"
    previews = data / "example_imgs"
    previews.mkdir(parents=True)
    for frame in (8, 16):
        Image.new("RGB", (16, 12), (frame, 80, 120)).save(previews / f"{frame:08d}.jpg")
    path = data / "example.h5"
    with pd.HDFStore(path, "w") as store:
        store.put("descriptions", pd.DataFrame({"id": [8, 16], "description": ["resting", "walking"]}))
        store.put("features_des", pd.DataFrame([[1., 0.], [0., 1.]], index=[8, 16]))
        store.get_storer("features_des").attrs.embedding_identity = json.dumps(
            embedding_identity(mousegpt_config.load_config("EMBEDDING"), 2))
    monkeypatch.setattr(module, "available_sessions", lambda: ["example"])
    monkeypatch.setattr(module, "session_config", lambda name: runtime_config.session_config(name, data_dir=data))
    monkeypatch.setattr(module, "log_dir", str(tmp_path / "logs"))
    return module, data, tmp_path


def test_interface_builds_with_portable_dependencies_without_provider_calls(interface, monkeypatch):
    module, _, _ = interface
    import requests
    import httpx
    fail = Mock(side_effect=AssertionError("UI construction must not call a provider"))
    monkeypatch.setattr(requests, "post", fail)
    monkeypatch.setattr(httpx.Client, "send", fail)
    ui = module.build_interface()
    components = ui.get_config_file()["components"]
    assert any(component["type"] == "chatbot" for component in components)
    assert any(component["type"] == "image" for component in components)
    assert module.chat_model.openai_api_base == "https://agent.example/v1"
    fail.assert_not_called()


@pytest.mark.parametrize("mode", ["invoke", "handler"])
def test_agent_executes_search_and_returns_tool_result(interface, monkeypatch, mode):
    module, _, tmp_path = interface
    from langchain_core.messages import AIMessageChunk, ToolMessage
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_openai import ChatOpenAI
    from tools_function import provider

    config = module.load_session_config("example")
    config["workspace"] = str(tmp_path / "results")
    Path(config["workspace"]).mkdir()
    monkeypatch.setattr(module.tools, "tool_config", config)
    calls = []

    def generate(_self, messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            message = AIMessageChunk(content="", tool_call_chunks=[{
                "name": "tool_search", "args": json.dumps({"behavior_description": "walking"}),
                "id": "search-1", "index": 0, "type": "tool_call_chunk",
            }])
        else:
            assert any(isinstance(item, ToolMessage) and "00000016" in str(item.content) for item in messages)
            message = AIMessageChunk(content="Frame 00000016 is the closest match.")
        yield ChatGenerationChunk(message=message)

    async def generate_async(self, messages, **kwargs):
        for chunk in generate(self, messages, **kwargs):
            yield chunk

    monkeypatch.setattr(ChatOpenAI, "_stream", generate)
    monkeypatch.setattr(ChatOpenAI, "_astream", generate_async)
    post = Mock(return_value=Mock(status_code=200, json=lambda: {"data": [{"embedding": [0., 1.]}]}))
    monkeypatch.setattr(provider.requests, "post", post)
    if mode == "invoke":
        result = module.agent_executor.invoke({"input": "Find walking frames.", "chat_history": []})
        assert "00000016" in result["output"]
        assert result["intermediate_steps"][0][1]["top_5"][0] == "00000016"
    else:
        import asyncio

        async def interact():
            history = None
            async for history in module.MGPT_handler("Find walking frames.", [], "test-user", config):
                pass
            return history

        history = asyncio.run(interact())
        assert "00000016" in history[-1].content
        assert any("Used tool tool_search" in item.metadata.get("title", "") for item in history)
    assert len(calls) == 2
    post.assert_called_once()
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer embedding-test-key"
    assert list(Path(config["workspace"]).glob("search_*.csv"))
