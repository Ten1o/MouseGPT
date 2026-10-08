"""Bounded HTTP requests shared by the natural-language analysis tools."""
import math
import time

import requests


def post_json(url, *, api_key, payload, validate, max_attempts=3, timeout=(10, 60)):
    """Retry transport, transient HTTP and invalid-response errors at most three times."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            status = response.status_code
            if not 200 <= status < 300:
                if status not in (408, 429) and status < 500:
                    raise RuntimeError(f"Provider request failed (HTTP {status}); no retry was made.")
                raise requests.HTTPError(f"Transient provider error (HTTP {status}).")
            result = response.json()
            validate(result)
            return result
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as error:
            if attempt + 1 == max_attempts:
                raise RuntimeError(f"Provider request failed after {max_attempts} attempts.") from error
            time.sleep(min(2 ** attempt, 8))


def encode_string_to_feature(behavior_description, api_key):
    """Obtain a finite, nonzero embedding or raise an explicit provider error."""
    def validate(result):
        values = result["data"][0]["embedding"]
        if not isinstance(values, list) or not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ) or not any(values):
            raise ValueError("The provider did not return a finite, nonzero embedding.")

    result = post_json(
        "https://api.openai.com/v1/embeddings",
        api_key=api_key,
        payload={"input": behavior_description, "model": "text-embedding-3-large"},
        validate=validate,
    )
    return result["data"][0]["embedding"]
