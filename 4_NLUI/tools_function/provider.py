"""Bounded HTTP requests shared by the natural-language analysis tools."""
import math
import time
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mousegpt_config import chat_request_options, http_request_config, load_config


def post_json(url, *, api_key, payload, validate, headers=None,
              max_attempts=3, timeout=(10, 60)):
    """Retry transport, transient HTTP and invalid-response errors at most three times."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                url,
                headers=(headers or {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }),
                json=payload,
                timeout=timeout,
            )
            status = response.status_code
            if not 200 <= status < 300:
                if status not in (408, 429) and status < 500:
                    raise RuntimeError(
                        f"Provider request failed (HTTP {status}); no retry was made: "
                        f"{response.text[:1000]}")
                raise requests.HTTPError(f"Transient provider error (HTTP {status}).")
            result = response.json()
            validate(result)
            return result
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as error:
            if attempt + 1 == max_attempts:
                raise RuntimeError(f"Provider request failed after {max_attempts} attempts.") from error
            time.sleep(min(2 ** attempt, 8))


def post_chat_completion(payload, *, api_key=None, validate, config=None, **request_options):
    """Call the configured analysis service with its own role credentials."""
    config = config or load_config("ANALYSIS", overrides={"api_key": api_key} if api_key else None)
    if config.role != "ANALYSIS":
        raise ValueError("Analysis requests require an ANALYSIS configuration.")
    url, headers = http_request_config(config, "chat/completions")
    request_payload = dict(payload)
    token_budget = request_payload.pop("max_completion_tokens", request_payload.pop("max_tokens", 2000))
    controls = {name: request_payload.pop(name, None) for name in ("temperature", "seed", "top_p")}
    request_payload.update(chat_request_options(config, max_tokens=token_budget, **controls))
    if config.provider == "azure_openai":
        request_payload.pop("model", None)
    else:
        request_payload["model"] = config.model
    return post_json(
        url, api_key=api_key, payload=request_payload, validate=validate,
        headers=headers, **request_options)


def encode_string_to_feature(behavior_description, api_key=None, *, config=None):
    """Obtain a finite, nonzero embedding or raise an explicit provider error."""
    def validate(result):
        values = result["data"][0]["embedding"]
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ) or not any(values):
            raise ValueError("The provider did not return a finite, nonzero embedding.")

    config = config or load_config("EMBEDDING", overrides={"api_key": api_key} if api_key else None)
    if config.role != "EMBEDDING":
        raise ValueError("Embedding requests require an EMBEDDING configuration.")
    url, headers = http_request_config(config, "embeddings")
    payload = {"input": behavior_description}
    if config.provider != "azure_openai":
        payload["model"] = config.model
    result = post_json(
        url,
        api_key=api_key,
        payload=payload,
        validate=validate,
        headers=headers,
    )
    return result["data"][0]["embedding"]
