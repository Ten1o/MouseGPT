"""Run inference using per-image annotation logs and save annotated previews."""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mousegpt_config import read_env_file


def insert_newlines(text, interval=60):
    result = []
    count = 0
    for char in text:
        result.append(char)
        count = 0 if char == "\n" else count + 1
        if count == interval:
            result.append("\n")
            count = 0
    return "".join(result)


def put_multiline_text(img, text, position, font, font_scale, color, thickness):
    import cv2

    x, y = position
    for index, line in enumerate(text.split("\n")):
        line_y = y + index * int(font_scale * 25 + thickness)
        img = cv2.putText(img, line, (x, line_y), font, font_scale, color, thickness)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-list", required=True, type=Path,
                        help="One image path per line, relative to this list's directory")
    parser.add_argument("--log-folder", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=(os.environ.get("MOUSEGPT_SERVING_API_KEY")
                                             or read_env_file().get("MOUSEGPT_SERVING_API_KEY")
                                             or "EMPTY"))
    parser.add_argument("--model", help="Served model ID; discovered from the endpoint when omitted")
    args = parser.parse_args()

    import cv2
    from openai import OpenAI
    from tqdm import tqdm

    image_list = args.image_list.resolve()
    images = [Path(line.strip()) for line in image_list.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    images = [path if path.is_absolute() else image_list.parent / path for path in images]
    if not images:
        raise ValueError("The image list is empty")
    if len({path.stem for path in images}) != len(images):
        raise ValueError("Image names must have unique stems so outputs cannot overwrite one another")
    for path in images:
        if not path.is_file() or not (args.log_folder / f"{path.stem}.json").is_file():
            raise FileNotFoundError(f"Missing image or annotation log for {path}")
    description_folder = args.output_folder / "output_des"
    preview_folder = args.output_folder / "output_img"
    description_folder.mkdir(parents=True, exist_ok=True)
    preview_folder.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=120, max_retries=2)
    model = args.model or client.models.list().data[0].id
    print(f"Model: {model}")

    for image_path in tqdm(images):
        log = json.loads((args.log_folder / f"{image_path.stem}.json").read_text(encoding="utf-8"))
        query = log["messages"][0]["content"] + "\n" + log["messages"][1]["content"][0]["text"]
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": query}], temperature=0,
            extra_body={"images": [str(image_path)]})
        text = response.choices[0].message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty model response for {image_path.name}")
        (description_folder / f"{image_path.stem}.txt").write_text(text, encoding="utf-8")
        try:
            text_on_img = json.dumps(json.loads(text), indent=4)
        except json.JSONDecodeError:
            text_on_img = text
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        annotated = put_multiline_text(img, insert_newlines(text_on_img), (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 255, 128), 2)
        if not cv2.imwrite(str(preview_folder / image_path.name), annotated):
            raise OSError(f"Failed to write preview for {image_path.name}")


if __name__ == "__main__":
    main()
