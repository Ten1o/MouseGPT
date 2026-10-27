"""Generate validated description embeddings with bounded API requests."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import tempfile
import time
import sys

import numpy as np
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mousegpt_config import http_request_config, load_config

from create_h5 import description_paths_by_id, description_text, load_data_matrix, read_description


EMBEDDING_MODEL = "text-embedding-3-large"


def resolve_embedding_api(api_key):
    config = load_config("EMBEDDING", overrides={"api_key": api_key} if api_key else None)
    url, headers = http_request_config(config, "embeddings")
    identity = {"provider": config.provider, "endpoint": config.endpoint,
                "model": config.model, "api_version": config.api_version}
    return url, headers, identity


def get_embeddings(content, api_key, *, max_attempts=3, retry_delay=1.0, timeout=30.0):
    if max_attempts < 1 or retry_delay < 0 or timeout <= 0:
        raise ValueError("Use positive max_attempts and timeout, and nonnegative retry_delay.")
    url, headers, api_identity = resolve_embedding_api(api_key)
    payload = {"input": content}
    if api_identity["provider"] != "azure_openai":
        payload["model"] = api_identity["model"]
    error = None
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout,
            )
            if 400 <= response.status_code < 500 and response.status_code not in (408, 429):
                raise RuntimeError(
                    f"Embedding request rejected with HTTP {response.status_code}: "
                    f"{response.text[:1000]}")
            response.raise_for_status()
            vector = np.asarray(response.json()["data"][0]["embedding"], dtype=float)
            if vector.shape != (3072,) or not np.isfinite(vector).all():
                raise ValueError("Provider returned an invalid 3072-element embedding.")
            return vector
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            error = exc
            if attempt + 1 < max_attempts:
                time.sleep(retry_delay)
    raise RuntimeError(f"Embedding request failed after {max_attempts} attempts.") from error


def process_file(filename, folder_path, out_folder, api_key, **request_options):
    source = Path(folder_path) / filename
    content = description_text(read_description(source))
    output = Path(out_folder) / f"{source.stem}_des.csv"
    metadata_path = Path(out_folder) / ".embedding-cache" / f"{source.stem}.json"
    _, _, api_identity = resolve_embedding_api(api_key)
    request_json = json.dumps(
        {"api": api_identity, "input": content, "model": api_identity["model"]},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    if output.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (metadata.get("request_sha256") == request_hash
                    and metadata.get("model") == api_identity["model"]
                    and metadata.get("vector_sha256") == hashlib.sha256(output.read_bytes()).hexdigest()):
                load_data_matrix([output])
                return output
        except (ValueError, TypeError, AttributeError):
            pass
    vector = get_embeddings(content, api_key, **request_options)
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{source.stem}.", suffix=".csv", dir=out_folder)
    os.close(handle)
    temporary_metadata = None
    try:
        np.savetxt(temporary, vector, delimiter=",")
        metadata = {
            "model": api_identity["model"],
            "api": api_identity,
            "request_sha256": request_hash,
            "vector_sha256": hashlib.sha256(Path(temporary).read_bytes()).hexdigest(),
        }
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=metadata_path.parent, delete=False) as handle:
            temporary_metadata = Path(handle.name)
            json.dump(metadata, handle, indent=2)
        os.replace(temporary, output)
        os.replace(temporary_metadata, metadata_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
        if temporary_metadata is not None:
            temporary_metadata.unlink(missing_ok=True)
    return output


def process_file_txt(filename, folder_path, out_folder, api_key, **request_options):
    return process_file(filename, folder_path, out_folder, api_key, **request_options)


def process_directory(data_folder, api_key, *, max_workers=8, **request_options):
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")
    descriptions = Path(data_folder) / "descriptions"
    paths = description_paths_by_id(descriptions)
    for path in paths.values():
        read_description(path)
    output = Path(data_folder) / "feature"
    output.mkdir(exist_ok=True)
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, path.name, descriptions, output, api_key, **request_options): path for path in paths.values()}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{futures[future].name}: {exc}")
    if errors:
        raise RuntimeError(f"Embedding generation failed for {len(errors)} file(s): " + "; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", required=True)
    parser.add_argument("--api", help="Optional key override; defaults to repository-root .env.local")
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--retry_delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    api_key = load_config("EMBEDDING", overrides={"api_key": args.api} if args.api else None).api_key
    process_directory(args.data_folder, api_key, max_workers=args.max_workers, max_attempts=args.max_attempts, retry_delay=args.retry_delay, timeout=args.timeout)


if __name__ == "__main__":
    main()
