"""Bounded requests and strict JSON parsing for behavior-analysis tools."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mousegpt_config import load_config


def parse_json_object(content):
    if not isinstance(content, str):
        raise ValueError("Expected a JSON response string.")
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Incomplete fenced JSON response.")
        content = "\n".join(lines[1:-1])
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def request_json(client, prompt, system_prompt, validator, *, max_attempts=3, model=None):
    if model is None:
        model = load_config("ANALYSIS", require_key=False).model
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")
    error = None
    for _ in range(max_attempts):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                timeout=60,
            )
            return validator(parse_json_object(completion.choices[0].message.content))
        except Exception as exc:
            error = exc
    raise RuntimeError(f"No valid model response after {max_attempts} attempts.") from error
