"""Run JSONL image prompts through a hosted VLM API or MouseGPT SWIFT.

The default SWIFT --validate-only path uses only the standard library. Hosted
API image validation also uses Pillow and the lightweight API dependencies.
Neither validation path needs a GPU or a local model checkpoint.
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import textwrap
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mousegpt_config import chat_request_options, http_request_config, load_config, read_env_file

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
            cases.append({"query": query, "images": paths,
                          "frame_id": record.get("frame_id", paths[0].stem)})
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


def annotation_helpers():
    """Load request/validation code only when inference starts."""
    sys.path.insert(0, str(HERE.parent / "data_prepareation"))
    import prompt_request
    return prompt_request


def prepare_api_cases(cases, config, max_tokens):
    """Validate every frame and image before scheduling a billable request."""
    from PIL import Image, ImageOps

    if config.role != "VISION":
        raise ValueError("Hosted image inference requires the VISION configuration.")
    annotation = annotation_helpers()
    prepared, used = [], set()
    for case in cases:
        raw_id = case.get("frame_id", case["images"][0].stem)
        if isinstance(raw_id, bool) or not str(raw_id).isascii() or not str(raw_id).isdigit():
            raise ValueError("API inputs need a nonnegative numeric frame_id or numeric first-image filename.")
        frame_id = str(int(raw_id)).zfill(8)
        if frame_id in used:
            raise ValueError(f"Duplicate frame ID {frame_id}; use one JSONL record per frame.")
        used.add(frame_id)
        content = [{"type": "text", "text": case["query"]}]
        preview = None
        image_sources = []
        for path in case["images"]:
            path = Path(path)
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError(f"API images must be JPEG or PNG: {path.name}")
                image.load()
                mime = "image/png" if image.format == "PNG" else "image/jpeg"
                if preview is None:
                    output = io.BytesIO()
                    ImageOps.exif_transpose(image).convert("RGB").save(output, format="JPEG", quality=95)
                    preview = output.getvalue()
            encoded = base64.b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            image_sources.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest()})
        payload = {
            "messages": [{"role": "system", "content": annotation.build_system_prompt()},
                         {"role": "user", "content": content}],
            **chat_request_options(config, max_tokens=max_tokens, temperature=0),
        }
        if config.provider != "azure_openai":
            payload["model"] = config.model
        prepared.append({"frame_id": frame_id, "payload": payload, "preview": preview,
                         "request_sha256": annotation.json_digest(payload), "images": image_sources})
    return prepared


def run_api_cases(cases, output_dir, config, *, max_attempts=3, timeout=120,
                  max_tokens=2000):
    """Write aligned descriptions and previews; resume only verified matches."""
    if max_attempts < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("max_attempts and timeout must be finite and positive.")
    annotation = annotation_helpers()
    prepared = prepare_api_cases(cases, config, max_tokens)
    if not prepared:
        raise ValueError("At least one input frame is required.")
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    identity = {"provider": config.provider, "endpoint": config.endpoint,
                "model": config.model, "api_version": config.api_version}
    contract = {"format_version": 1, "api": identity, "frames": [
        {key: case[key] for key in ("frame_id", "request_sha256", "images")}
        for case in prepared]}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict)
                or any(manifest.get(key) != value for key, value in contract.items())
                or not isinstance(manifest.get("completed"), dict)):
            raise ValueError("Output belongs to different inputs/model or has invalid provenance; use a new --output-dir.")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Output directory contains files without matching provenance; use a new --output-dir.")
        root.mkdir(parents=True, exist_ok=True)
        manifest = {**contract, "completed": {}, "status": "in_progress"}
        annotation.write_json(manifest_path, manifest)

    # Verify every existing result before making any new request.
    def result_paths(frame_id):
        return {"description": root / "descriptions" / f"{frame_id}.json",
                "raw_response": root / "raw_descriptions" / f"{frame_id}.json",
                "preview": root / "images" / f"{frame_id}.jpg"}

    if set(manifest["completed"]) - {case["frame_id"] for case in prepared}:
        raise ValueError("Unexpected completed frame IDs in manifest; use a new --output-dir.")
    for case in prepared:
        frame_id = case["frame_id"]
        paths = result_paths(frame_id)
        if frame_id in manifest["completed"]:
            hashes = manifest["completed"][frame_id]
            if not isinstance(hashes, dict) or any(
                not path.is_file() or path.is_symlink()
                or hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(name)
                for name, path in paths.items()
            ):
                raise ValueError(f"Saved outputs for {frame_id} were changed or are incomplete; use a new --output-dir.")
            response = json.loads(paths["raw_response"].read_text(encoding="utf-8"))
            if annotation.parse_description(response) != json.loads(paths["description"].read_text(encoding="utf-8")):
                raise ValueError(f"Description does not match saved response for {frame_id}.")
        elif any(path.exists() for path in paths.values()):
            raise FileExistsError(f"Unverified partial outputs for {frame_id}; use a new --output-dir.")

    url, headers = http_request_config(config, "chat/completions")
    for index, case in enumerate(prepared, 1):
        frame_id = case["frame_id"]
        if frame_id in manifest["completed"]:
            print(f"[{index}/{len(prepared)}] {frame_id} (verified cached output)")
            continue
        try:
            response = annotation.request_description(case["payload"], url, headers,
                                                       max_attempts, timeout=timeout)
            try:
                description = annotation.parse_description(response)
            except ValueError as error:
                failed = root / "failed_responses" / f"{frame_id}.{uuid.uuid4().hex}.json"
                annotation.write_json(failed, response)
                raise ValueError(f"Invalid description for {frame_id}; response saved at {failed}: {error}") from error
            paths = result_paths(frame_id)
            annotation.write_json(paths["raw_response"], response)
            annotation.write_json(paths["description"], description)
            paths["preview"].parent.mkdir(parents=True, exist_ok=True)
            with paths["preview"].open("xb") as handle:
                handle.write(case["preview"])
            manifest["completed"][frame_id] = {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in paths.items()}
            annotation.write_json(manifest_path, manifest)
        except Exception:
            manifest["status"] = "incomplete"
            annotation.write_json(manifest_path, manifest)
            raise
        print(f"[{index}/{len(prepared)}] {frame_id}")
    manifest["status"] = "complete"
    annotation.write_json(manifest_path, manifest)
    print(f"Validated {len(prepared)} descriptions and matching previews in {root}.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("swift", "api"), default="swift",
                        help="api uses the configured VISION service; swift uses a local model server")
    parser.add_argument("--input", type=Path, default=HERE / "data/test_data.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--base-url", default=os.environ.get("MOUSEGPT_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", help="Served model ID; auto-select only when exactly one is available")
    parser.add_argument("--api-key-env", default="MOUSEGPT_SERVING_API_KEY", help="Environment variable containing the SWIFT server API key")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request read timeout in seconds")
    parser.add_argument("--max-attempts", type=int, default=3, help="API backend: maximum attempts per frame for transient errors")
    parser.add_argument("--max-tokens", type=int, default=2000, help="API backend: completion token budget per frame, including reasoning tokens where applicable")
    parser.add_argument("--limit", type=int, help="Run only the first N records")
    parser.add_argument("--save-previews", action="store_true", help="Save annotated first-image previews; requires opencv-python")
    parser.add_argument("--validate-only", action="store_true", help="Check JSONL and image paths without contacting a model")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.max_attempts < 1 or args.max_tokens < 1:
        parser.error("--max-attempts and --max-tokens must be positive")
    try:
        cases = load_cases(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.validate_only:
        if args.backend == "api":
            # Validate images, IDs and payload construction without requiring a key.
            try:
                config = load_config("VISION", overrides={"model": args.model} if args.model else None,
                                     require_key=False)
                prepare_api_cases(cases, config, args.max_tokens)
            except (OSError, ValueError) as error:
                parser.error(str(error))
        print(f"Validated {len(cases)} prompts and {sum(len(case['images']) for case in cases)} image references.")
        return 0

    if args.backend == "api":
        try:
            config = load_config("VISION", overrides={"model": args.model} if args.model else None)
            return run_api_cases(cases, args.output_dir, config, max_attempts=args.max_attempts,
                                 timeout=args.timeout, max_tokens=args.max_tokens)
        except (OSError, ValueError, RuntimeError) as error:
            parser.exit(1, f"VLM API error: {error}\n")

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
