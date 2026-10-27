"""Make at most one small model-service request for one configured role."""

import argparse
import base64
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mousegpt_config import http_request_config, load_config


SAMPLE_IMAGE = ROOT / "2_VLM" / "prompt_test" / "data" / "imgs" / "00001344.jpg"
ROLES = ("AGENT", "ANALYSIS", "EMBEDDING", "VISION")


def request_body(config, image_path=None):
    if config.role == "EMBEDDING":
        return "embeddings", {"model": config.model, "input": "mouse walking"}
    user_content = "Reply with exactly OK."
    if config.role == "VISION":
        image = Path(image_path or SAMPLE_IMAGE)
        if not image.is_file():
            raise FileNotFoundError(f"Sample image not found: {image}")
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError("Vision sample must be a JPEG or PNG file.")
        if image.stat().st_size > 1_000_000:
            raise ValueError("Vision sample must be at most 1 MB for this smoke test.")
        mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        user_content = [
            {"type": "text", "text": "Name the animal in this image in one short sentence."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    token_field = "max_completion_tokens" if config.provider == "azure_openai" else "max_tokens"
    body = {"model": config.model,
            "messages": [{"role": "user", "content": user_content}],
            token_field: 256}
    if config.provider == "deepseek":
        body["thinking"] = {"type": "disabled"}
    return "chat/completions", body


def run(role, *, live=False, image=None):
    config = load_config(role)
    operation, body = request_body(config, image)
    url, headers = http_request_config(config, operation)
    print(f"Role: {config.role}; provider: {config.provider}; model: {config.model}")
    print(f"Operation: {operation}; endpoint: {config.endpoint}")
    if not live:
        print("Dry run only. Add --live to send exactly one request.")
        return 0
    import requests

    try:
        response = requests.post(url, headers=headers, json=body, timeout=(10, 60))
    except requests.RequestException as error:
        print(f"Network request failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"HTTP status: {response.status_code}")
    if not response.ok:
        # Provider responses can contain account details; avoid echoing them.
        print("Provider rejected the request. Check deployment, API version, key permissions, and network.",
              file=sys.stderr)
        return 1
    try:
        result = response.json()
        if role == "EMBEDDING":
            values = result["data"][0]["embedding"]
            print(f"Embedding dimensions: {len(values)}")
            if len(values) != 3072:
                print("MouseGPT Stage 3 expects 3072 dimensions.", file=sys.stderr)
                return 1
        else:
            message = result["choices"][0]["message"].get("content")
            if not isinstance(message, str) or not message.strip():
                print("Request was accepted, but it returned no assistant text. This is inconclusive; "
                      "check model response limits or deployment behavior.", file=sys.stderr)
                return 1
            print(f"Assistant response: {message[:300]}")
        usage = result.get("usage")
        if usage:
            print(f"Reported usage: {usage}")
    except (ValueError, KeyError, IndexError, TypeError) as error:
        print(f"Unexpected response shape: {type(error).__name__}", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, type=str.upper, choices=ROLES)
    parser.add_argument("--image", type=Path, help="JPEG/PNG for the VISION request; defaults to a bundled sample")
    parser.add_argument("--live", action="store_true", help="Send one potentially billable request")
    args = parser.parse_args(argv)
    if args.image and args.role != "VISION":
        parser.error("--image is only valid for VISION")
    try:
        return run(args.role, live=args.live, image=args.image)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
