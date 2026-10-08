"""Generate structured descriptions from synchronized camera images."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import math
import os
from pathlib import Path
import tempfile
import time

import requests
from tqdm import tqdm


DESCRIPTION_FIELDS = ("Overall", "Head", "Limb", "Torso", "Others", "Keywords")


def build_system_prompt():
    system_prompt = "The user will show you one frame of a photo or several photos of a mouse in a glass chamber shot at different views at the same time, and necessary information such as the absolute value of velocity, etc.. "

    system_prompt += "You need to give me the description of the motion of the mice in the photos. You should give me a .json format, where there are several elements. You should describe the motion of the mouse in each part of json using full sentences as detailed as possible. You should answer each of the following questions. Your response should include but not limited to information of below questions. "

    system_prompt += "The first element in the json is \"Overall\". From the images, what is the overall status of the mouse, stationary or in motion? If it is moving, the movement is slow, moderate or rapid? What is this mouse doing? Is it standing, walking, or doing nothing?\n"

    system_prompt += "Next element: \"Head\". It should be the description of the head and face region. What can you see from its face? Is it just looking around? Or is it licking / grooming its paw or fur? What is the head position and orientation with respect to its body or the ground? Is the head turning left, turning right or upward? Which direction is it looking into? \n"

    system_prompt += "Next element: \"Limb\". What is its front paw / front foot doing? Is it on the ground or in the air? What is its hind paw / hind leg doing? Is it on the ground or in the air? Is it clearly walking? What is its tail doing? Is it extended on the ground, in the air parallel to the ground or pointed upwards?\n"

    system_prompt += "Next element: \"Torso\". What can you see from its main body, or torso? Is the spine curved vertically, horizontally, or stretched? Are the chest, abdomen and hip (or butt) partially contacted with the ground, fully lay on the ground, or in the air? \n"

    system_prompt += "Next element: \"Others\". What else can you directly see about the motion or behavior of this mouse in this photo? Please include other infomation and describe it as detailed as possible, but still the description should be facts, rather than opinions or assumptions. \n"

    system_prompt += "And the last element would be the \"Keywords\". You should use line of keywords separated by commas to conclude and summarize the identifiable features of the mouse behavior in each of the above components.\n"

    system_prompt += "Note that your description / keywords should only be about the animal itself, without the environments. And your answers should only contain facts, without opinions or assumptions. Do NOT use negative sentence such as \"rather than\", or \"instead of\". The main part of the description should come from the image, and the speed data is only used as an auxiliary. The velocity is an absolute value without direction, so a high velocity does not necessarily mean the mouse is walking or running, it might just caused by vertical movement of the body, such as standing up. And the value of velocity is only an estimate with noise, so a low value such as 1 cm/s might be only caused by noise or shaking of the mouse, instead of the actual movement, a high value might also caused by a detection failure. Thus, the major conclusion should only come from the images themselves, and the velocity is only a cross-validation. Also, do NOT include exact number of speed in your respond, use natural language to describe it instead. \n"
    
    return system_prompt


def parse_description(response):
    """Extract a JSON object from a Chat Completions response, including JSON fences."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Response has no assistant message content") from error
    if not isinstance(content, str):
        raise ValueError("Assistant message content must be text")
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].strip().lower() not in ("```json", "```") or lines[-1].strip() != "```":
            raise ValueError("Invalid JSON code fence")
        content = "\n".join(lines[1:-1])
    description = json.loads(content)
    if not isinstance(description, dict):
        raise ValueError("Description must be a JSON object")
    for field in DESCRIPTION_FIELDS:
        if not isinstance(description.get(field), str) or not description[field].strip():
            raise ValueError(f"Description requires a nonempty text field: {field}")
    return description


def write_json(path, value):
    """Replace a JSON file only after serialization and writing succeed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fix_json_files(data_folder):
    """Convert saved API envelopes into validated descriptions without another API call."""
    root = Path(data_folder)
    count = 0
    for source in sorted((root / "raw_descriptions").glob("*.json")):
        try:
            description = parse_description(json.loads(source.read_text(encoding="utf-8")))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid description in {source}: {error}") from error
        write_json(root / "descriptions" / source.name, description)
        count += 1
    return count


def paired_images(data_folder):
    root = Path(data_folder) / "pre_images"
    cameras = []
    for camera in (3, 5):
        folder = root / f"data{camera}"
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        images = {}
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if path.stem in images:
                raise ValueError(f"Duplicate frame ID {path.stem} in {folder}")
            int(path.stem)
            images[path.stem] = path
        cameras.append(images)
    if not cameras[0] or cameras[0].keys() != cameras[1].keys():
        raise ValueError("Camera 3 and camera 5 must contain the same nonempty set of frame IDs")
    return [(cameras[0][frame], cameras[1][frame]) for frame in sorted(cameras[0])]


def request_description(payload, api_key, max_attempts=3):
    """Retry only transient transport/server errors within a fixed attempt limit."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    for attempt in range(max_attempts):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"}, json=payload,
                timeout=(10, 120),
            )
        except (requests.Timeout, requests.ConnectionError):
            if attempt + 1 == max_attempts:
                raise
        else:
            if response.status_code == 200:
                return response.json()
            transient = response.status_code == 429 or 500 <= response.status_code < 600
            if not transient or attempt + 1 == max_attempts:
                raise RuntimeError(f"Description API returned HTTP {response.status_code}")
        time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("Description API attempts exhausted")


def json_digest(value):
    """Hash canonical JSON without including authentication headers or API keys."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def process_images(file1, file2, data_folder, velocity_list, api_key, max_attempts=3,
                   model="gpt-4o-2024-05-13"):
    file1, file2 = Path(file1), Path(file2)
    root = Path(data_folder)
    output_basename = f"{file1.stem}.json"
    raw_path = root / "raw_descriptions" / output_basename
    description_path = root / "descriptions" / output_basename
    frame = str(int(file1.stem))
    velocity = velocity_list[frame]
    prompt = "The first two photos shot at different view at the same time. "
    prompt += "The overall movement speed of the mouse is: {:.2f} cm/s, the speed of the head: {:.2f} cm/s, speed of left and right forelimbs: {:.2f}, {:.2f} cm/s, and left and right hind limbs are {:.2f}, {:.2f} cm/s respectively.".format(
        velocity["base"], velocity["head"], velocity["forepawL"], velocity["forepawR"],
        velocity["hindpawL"], velocity["hindpawR"],
    )
    content = [{"type": "text", "text": prompt}]
    for path in (file1, file2):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{encoded}", "detail": "high"}})
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": build_system_prompt()},
                     {"role": "user", "content": content}],
        "seed": 1, "temperature": 0, "max_tokens": 2000,
    }
    log_path = root / "log" / output_basename
    request_hash = json_digest(payload)
    if raw_path.exists() and log_path.exists():
        try:
            cached = json.loads(raw_path.read_text(encoding="utf-8"))
            log = json.loads(log_path.read_text(encoding="utf-8"))
            provenance = log.get("_mousegpt_cache", {})
            if (provenance.get("request_sha256") == request_hash
                    and provenance.get("response_sha256") == json_digest(cached)):
                description = parse_description(cached)
                write_json(description_path, description)
                return description_path
        except (ValueError, TypeError, AttributeError):
            pass
    response = request_description(payload, api_key, max_attempts)
    # Bind both files to the same request and response, including interrupted writes.
    write_json(raw_path, response)
    write_json(log_path, {**payload, "_mousegpt_cache": {
        "request_sha256": request_hash, "response_sha256": json_digest(response),
    }})
    description = parse_description(response)
    write_json(description_path, description)
    return description_path


def run_requests(data_folder, api_key, max_attempts=3, workers=10,
                 model="gpt-4o-2024-05-13"):
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or supply --api")
    if max_attempts < 1 or workers < 1:
        raise ValueError("max_attempts and workers must be at least 1")
    pairs = paired_images(data_folder)
    velocity_list = json.loads((Path(data_folder) / "velocity.json").read_text(encoding="utf-8"))
    # Check every frame before scheduling any billable requests.
    for file1, _ in pairs:
        velocity = velocity_list[str(int(file1.stem))]
        for field in ("base", "head", "forepawL", "forepawR", "hindpawL", "hindpawR"):
            value = velocity[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Velocity {field} for {file1.stem} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Velocity {field} for {file1.stem} must be finite and nonnegative")
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_images, first, second, data_folder,
                                   velocity_list, api_key, max_attempts, model): first.stem
                   for first, second in pairs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as error:
                failures.append(f"{futures[future]}: {error}")
    if failures:
        raise RuntimeError(f"{len(failures)} frame(s) failed: " + "; ".join(failures))
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", required=True, help="Folder containing pre_images and velocity.json")
    parser.add_argument("--api", default=os.environ.get("OPENAI_API_KEY"), help="API key; defaults to OPENAI_API_KEY")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum attempts per frame for transient errors")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent image-pair requests")
    parser.add_argument("--model", default="gpt-4o-2024-05-13", help="Model identifier")
    args = parser.parse_args()
    run_requests(args.data_folder, args.api, args.max_attempts, args.workers, args.model)


if __name__ == "__main__":
    main()
