"""Read behavioral descriptions consistently without scientific dependencies."""
import hashlib
import json
from pathlib import Path


def parse_description_text(content):
    """Read plain text or a JSON object, including a fenced model response."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Description must be nonempty text or a JSON object.")
    content = content.strip()
    if "{" not in content:
        if content.startswith("```") or content.endswith("}"):
            raise ValueError("Malformed JSON description.")
        return content
    start = content.index("{")
    try:
        value, end = json.JSONDecoder().raw_decode(content[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Malformed JSON description.") from exc
    trailing = content[start + end:].strip()
    if trailing not in ("", "```"):
        raise ValueError("Unexpected content after description JSON.")
    if not isinstance(value, dict) or not value:
        raise ValueError("Description JSON must be a nonempty object.")
    return value


def description_text(description):
    """Use descriptive fields for JSON and text inputs, excluding keywords."""
    if isinstance(description, dict):
        values = [value for key, value in description.items() if key != "Keywords"]
        if not all(isinstance(value, str) for value in values):
            raise ValueError("Description fields must contain strings.")
        content = " ".join(value.strip() for value in values if value.strip())
    elif isinstance(description, str):
        content = description.strip()
    else:
        raise ValueError("Description must be text or an object.")
    if not content:
        raise ValueError("Description has no text to embed.")
    return content


def read_description_snapshot(path, *, expected_sha256=None):
    """Parse and fingerprint the same bytes; optionally require a prior hash."""
    path = Path(path)
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and fingerprint != expected_sha256:
        raise ValueError("Source description changed. Reload it before saving the review.")
    content = raw.decode("utf-8")
    if path.suffix.lower() == ".txt":
        value = parse_description_text(content)
    else:
        try:
            value = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}") from exc
        if isinstance(value, dict) and "choices" in value:
            choices = value["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"No valid description in {path}")
            value = parse_description_text(choices[0].get("message", {}).get("content"))
        if not isinstance(value, (str, dict)):
            raise ValueError(f"Invalid description in {path}")
    description_text(value)
    return value, fingerprint


def read_description(path, *, expected_sha256=None):
    return read_description_snapshot(path, expected_sha256=expected_sha256)[0]
