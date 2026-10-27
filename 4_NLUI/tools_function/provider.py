"""Bounded HTTP requests shared by the natural-language analysis tools."""
import math
import time
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mousegpt_config import http_request_config, load_config


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


def post_chat_completion(payload, *, api_key, validate, **request_options):
    """Call either OpenAI or the configured Azure chat deployment."""
    config = load_config("ANALYSIS", overrides={"api_key": api_key} if api_key else None)
    url, headers = http_request_config(config, "chat/completions")
    request_payload = dict(payload)
    if config.provider == "azure_openai":
        request_payload.pop("model", None)
        if "max_tokens" in request_payload:
            request_payload["max_completion_tokens"] = request_payload.pop("max_tokens")
        # The configured Azure reasoning deployment may reject sampling controls.
        request_payload.pop("temperature", None)
        request_payload.pop("top_p", None)
    else:
        request_payload["model"] = config.model
        if config.provider != "openai":
            request_payload.pop("temperature", None)
            request_payload.pop("top_p", None)
    return post_json(
        url, api_key=api_key, payload=request_payload, validate=validate,
        headers=headers, **request_options)


def encode_string_to_feature(behavior_description, api_key):
    """Obtain a finite, nonzero embedding or raise an explicit provider error."""
    def validate(result):
        values = result["data"][0]["embedding"]
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ) or not any(values):
            raise ValueError("The provider did not return a finite, nonzero embedding.")

    config = load_config("EMBEDDING", overrides={"api_key": api_key} if api_key else None)
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
