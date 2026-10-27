"""Run JSONL image prompts against a MouseGPT SWIFT deployment.

Only standard-library modules are imported until inference starts, so --help
and --validate-only work without a GPU or the model dependencies.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import textwrap
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mousegpt_config import read_env_file

HERE = Path(__file__).resolve().parent


def load_cases(jsonl_path):
    """Resolve each local image path relative to its JSONL file, not the CWD."""
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    cases = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                query = record["query"]
                images = record["images"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{jsonl_path}:{line_number}: invalid prompt record") from error
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Line {line_number}: query must be a nonempty string")
            if not isinstance(images, list) or not images:
                raise ValueError(f"Line {line_number}: images must be a nonempty list")
            paths = []
            for value in images:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Line {line_number}: image paths must be strings")
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = jsonl_path.parent / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Line {line_number}: missing image {path}")
                paths.append(path)
            cases.append({"query": query, "images": paths})
    if not cases:
        raise ValueError(f"No prompt records found in {jsonl_path}")
    return cases


def encode_images(paths):
    # The bundled SWIFT vision loader accepts raw base64 strings. Sending bytes
    # avoids requiring the deployment server to share the client's filesystem.
    return [base64.b64encode(path.read_bytes()).decode("ascii") for path in paths]


def save_preview(image_path, response_text, destination):
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    try:
        response_text = json.dumps(json.loads(response_text), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    lines = []
    for paragraph in response_text.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=60) or [""])
    for index, line in enumerate(lines):
        cv2.putText(image, line, (10, 30 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (128, 255, 128), 2)
    if not cv2.imwrite(str(destination), image):
        raise OSError(f"Could not write preview: {destination}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data/test_data.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--base-url", default=os.environ.get("MOUSEGPT_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", help="Served model ID; auto-select only when exactly one is available")
    parser.add_argument("--api-key-env", default="MOUSEGPT_SERVING_API_KEY", help="Environment variable containing the SWIFT server API key")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    parser.add_argument("--limit", type=int, help="Run only the first N records")
    parser.add_argument("--save-previews", action="store_true", help="Save annotated first-image previews; requires opencv-python")
    parser.add_argument("--validate-only", action="store_true", help="Check JSONL and image paths without contacting a model")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        cases = load_cases(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.validate_only:
        print(f"Validated {len(cases)} prompts and {sum(len(case['images']) for case in cases)} image references.")
        return 0

    from openai import OpenAI

    serving_key = os.environ.get(args.api_key_env) or read_env_file().get(args.api_key_env) or "EMPTY"
    client = OpenAI(api_key=serving_key,
                    base_url=args.base_url, timeout=args.timeout)
    model = args.model
    if model is None:
        models = client.models.list().data
        if len(models) != 1:
            parser.error("Specify --model: the endpoint did not expose exactly one model")
        model = models[0].id
    descriptions = args.output_dir / "output_des"
    descriptions.mkdir(parents=True, exist_ok=True)
    previews = args.output_dir / "output_img"
    if args.save_previews:
        previews.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(cases, 1):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": case["query"]}],
            temperature=0,
            extra_body={"images": encode_images(case["images"])},
        )
        response_text = response.choices[0].message.content
        if not isinstance(response_text, str):
            raise ValueError(f"Prompt {index}: endpoint returned no text response")
        # Prefix indices to avoid collisions when different folders reuse names.
        stem = f"{index:04d}_{case['images'][0].stem}"
        (descriptions / f"{stem}.txt").write_text(response_text, encoding="utf-8")
        if args.save_previews:
            save_preview(case["images"][0], response_text, previews / f"{stem}.jpg")
        print(f"[{index}/{len(cases)}] {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
