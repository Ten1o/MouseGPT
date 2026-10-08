"""Offline regression checks for video extraction and annotation requests."""

import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
from unittest.mock import Mock, patch

import pytest
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / "2_VLM/data_prepareation" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


video = load_module("vids_to_imgs.py")
annotation = load_module("prompt_request.py")


def make_videos(tmp_path):
    source = tmp_path / "video input ; literal $(name)"
    source.mkdir()
    for camera in (3, 5):
        (source / f"data{camera}.mp4").write_bytes(b"video")
    return source


def emit_frames(command, **kwargs):
    assert kwargs == {"check": True}
    assert isinstance(command, list)
    pattern = Path(command[-1])
    for index in (1, 2, 3):
        pattern.with_name(f"{index:08d}.jpg").write_bytes(f"frame-{index}".encode())


def test_extraction_preserves_spacing_and_literal_paths(tmp_path):
    source = make_videos(tmp_path)
    output = tmp_path / "images output ; literal $(name)"
    with patch.object(video.subprocess, "run", side_effect=emit_frames) as run:
        video.extract_frames(source, output, 1, 2, 59.94)
    assert run.call_count == 2
    assert str(source / "data3.mp4") in run.call_args_list[0].args[0]
    for camera in (3, 5):
        folder = output / "images" / f"data{camera}"
        assert sorted(path.name for path in folder.iterdir()) == [
            "00003596.jpg", "00003608.jpg", "00003620.jpg"]
        assert (folder / "00003608.jpg").read_bytes() == b"frame-2"
    assert not list((output / "images").glob(".extract-*"))


def test_second_video_failure_does_not_publish_partial_results_or_delete_other_files(tmp_path):
    source = make_videos(tmp_path)
    output = tmp_path / "output"
    legacy_temp = output / "images" / "data3_temp"
    legacy_temp.mkdir(parents=True)
    marker = legacy_temp / "keep.txt"
    marker.write_text("keep")
    count = 0

    def fail_second(command, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise subprocess.CalledProcessError(1, command)
        emit_frames(command, **kwargs)

    with patch.object(video.subprocess, "run", side_effect=fail_second):
        with pytest.raises(subprocess.CalledProcessError):
            video.extract_frames(source, output, 0, 1, 60)
    assert marker.read_text() == "keep"
    assert not (output / "images/data3").exists()
    assert not (output / "images/data5").exists()
    assert not list((output / "images").glob(".extract-*"))


def test_existing_extraction_is_preserved_without_starting_ffmpeg(tmp_path):
    source = make_videos(tmp_path)
    output = tmp_path / "output"
    existing = output / "images/data3"
    existing.mkdir(parents=True)
    marker = existing / "00000000.jpg"
    marker.write_bytes(b"existing")
    with patch.object(video.subprocess, "run") as run:
        with pytest.raises(FileExistsError):
            video.extract_frames(source, output, 0, 1, 60)
    run.assert_not_called()
    assert marker.read_bytes() == b"existing"


@pytest.mark.parametrize("fps,start,end", [(1.05, 0, 0), (1.05, 1, 2),
                                          (1.05, 1, 0), (1.0, 1, 2)])
def test_real_ffmpeg_content_matches_original_frame_filenames(tmp_path, fps, start, end):
    """Encode each source frame's ordinal as a binary barcode and read it back."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    frame_count = 139
    source = tmp_path / "videos with spaces"
    source.mkdir()
    frames = np.zeros((frame_count, 16, 64, 3), dtype=np.uint8)
    for index in range(frame_count):
        for bit in range(8):
            frames[index, :, bit * 8:(bit + 1) * 8, :] = 255 * ((index >> bit) & 1)
    subprocess.run(
        [executable, "-v", "error", "-nostdin", "-n", "-f", "rawvideo",
         "-pixel_format", "rgb24", "-video_size", "64x16", "-framerate", str(fps),
         "-i", "pipe:0", "-an", "-c:v", "libx264rgb", "-crf", "0",
         "-preset", "ultrafast", str(source / "data3.mp4")],
        input=frames.tobytes(), check=True,
    )
    shutil.copyfile(source / "data3.mp4", source / "data5.mp4")
    output = tmp_path / "output with spaces"
    video.extract_frames(source, output, start, end, fps, ffmpeg=executable)
    first = int(fps * 60 * start)
    stop = int(fps * 60 * end) if end else frame_count
    expected = list(range(first, min(stop, frame_count), 12))
    for camera in (3, 5):
        images = sorted((output / "images" / f"data{camera}").glob("*.jpg"))
        assert [int(path.stem) for path in images] == expected
        for path in images:
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("L"))
            decoded = sum((int(pixels[8, bit * 8 + 4]) > 128) << bit for bit in range(8))
            assert decoded == int(path.stem)


def description_response(fenced=False):
    description = {field: f"Example {field}." for field in annotation.DESCRIPTION_FIELDS}
    content = json.dumps(description)
    if fenced:
        content = f"```json\n{content}\n```"
    return {"choices": [{"message": {"content": content}}]}, description


def annotation_input(tmp_path, frames=("00000000",)):
    for camera in (3, 5):
        folder = tmp_path / "pre_images" / f"data{camera}"
        folder.mkdir(parents=True)
        for frame in frames:
            (folder / f"{frame}.jpg").write_bytes(b"image")
    velocities = {str(int(frame)): {field: 0.0 for field in
                  ("base", "head", "forepawL", "forepawR", "hindpawL", "hindpawR")}
                  for frame in frames}
    (tmp_path / "velocity.json").write_text(json.dumps(velocities))
    return tmp_path


def test_json_conversion_produces_structured_description_without_external_helper(tmp_path):
    source = tmp_path / "raw_descriptions"
    source.mkdir()
    response, description = description_response(fenced=True)
    (source / "00000000.json").write_text(json.dumps(response))
    assert annotation.fix_json_files(tmp_path) == 1
    assert json.loads((tmp_path / "descriptions/00000000.json").read_text()) == description


def test_camera_pair_mismatch_is_detected_before_requests(tmp_path):
    root = annotation_input(tmp_path)
    (root / "pre_images/data5/00000000.jpg").rename(root / "pre_images/data5/00000012.jpg")
    with patch.object(annotation.requests, "post") as post:
        with pytest.raises(ValueError, match="same nonempty set"):
            annotation.run_requests(root, "test-key")
    post.assert_not_called()


def test_annotation_writes_parsed_output_and_resumes_saved_response_without_api(tmp_path):
    root = annotation_input(tmp_path)
    response, description = description_response(fenced=True)
    with patch.object(annotation.requests, "post", return_value=Mock(status_code=200, json=lambda: response)) as post:
        assert annotation.run_requests(root, "test-key", workers=1) == 1
        assert annotation.run_requests(root, "test-key", workers=1) == 1
    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == (10, 120)
    assert json.loads((root / "descriptions/00000000.json").read_text()) == description
    assert json.loads((root / "raw_descriptions/00000000.json").read_text()) == response


def test_transient_errors_have_finite_attempt_budget():
    with patch.object(annotation.requests, "post", return_value=Mock(status_code=429)) as post:
        with patch.object(annotation.time, "sleep"):
            with pytest.raises(RuntimeError, match="429"):
                annotation.request_description({}, "test-key", max_attempts=3)
    assert post.call_count == 3


def test_permanent_api_error_is_not_retried():
    with patch.object(annotation.requests, "post", return_value=Mock(status_code=401)) as post:
        with patch.object(annotation.time, "sleep") as sleep:
            with pytest.raises(RuntimeError, match="401"):
                annotation.request_description({}, "test-key", max_attempts=3)
    assert post.call_count == 1
    sleep.assert_not_called()


def test_invalid_response_is_visible_and_does_not_become_a_success_file(tmp_path):
    root = annotation_input(tmp_path)
    response = {"choices": [{"message": {"content": "not valid JSON"}}]}
    with patch.object(annotation.requests, "post", return_value=Mock(status_code=200, json=lambda: response)) as post:
        with pytest.raises(RuntimeError, match="00000000"):
            annotation.run_requests(root, "test-key", max_attempts=3, workers=1)
    assert post.call_count == 1
    assert (root / "raw_descriptions/00000000.json").exists()
    assert not (root / "descriptions/00000000.json").exists()


@pytest.mark.parametrize("changed", ["model", "velocity", "image", "prompt", "missing_log", "missing_hash", "mismatched_response"])
def test_annotation_cache_requires_matching_inputs_and_bound_response(tmp_path, changed):
    root = annotation_input(tmp_path)
    response, description = description_response()
    with patch.object(annotation.requests, "post", return_value=Mock(status_code=200, json=lambda: response)) as post:
        annotation.run_requests(root, "test-key", workers=1, model="model-a")
        model = "model-a"
        if changed == "model":
            model = "model-b"
        elif changed == "velocity":
            path = root / "velocity.json"
            velocities = json.loads(path.read_text())
            velocities["0"]["base"] = 100.0
            path.write_text(json.dumps(velocities))
        elif changed == "image":
            (root / "pre_images/data3/00000000.jpg").write_bytes(b"different image")
        elif changed == "missing_log":
            (root / "log/00000000.json").unlink()
        elif changed == "missing_hash":
            path = root / "log/00000000.json"
            log = json.loads(path.read_text())
            log.pop("_mousegpt_cache")
            path.write_text(json.dumps(log))
        elif changed == "mismatched_response":
            path = root / "raw_descriptions/00000000.json"
            cached = json.loads(path.read_text())
            cached["id"] = "different-response"
            path.write_text(json.dumps(cached))
        if changed == "prompt":
            with patch.object(annotation, "build_system_prompt", return_value="Updated prompt"):
                annotation.run_requests(root, "test-key", workers=1, model=model)
        else:
            annotation.run_requests(root, "test-key", workers=1, model=model)
    assert post.call_count == 2
    log = json.loads((root / "log/00000000.json").read_text())
    assert "test-key" not in json.dumps(log)
    assert len(log["_mousegpt_cache"]["request_sha256"]) == 64


def test_annotation_failed_regeneration_does_not_mark_stale_response_as_current(tmp_path):
    root = annotation_input(tmp_path)
    response, _ = description_response()
    with patch.object(annotation.requests, "post", return_value=Mock(status_code=200, json=lambda: response)):
        annotation.run_requests(root, "test-key", workers=1, model="old-model")
    with patch.object(annotation.requests, "post", return_value=Mock(status_code=401)):
        with pytest.raises(RuntimeError, match="401"):
            annotation.run_requests(root, "test-key", workers=1, model="new-model")
    with patch.object(annotation.requests, "post", return_value=Mock(status_code=200, json=lambda: response)) as post:
        annotation.run_requests(root, "test-key", workers=1, model="new-model")
    assert post.call_count == 1
    assert json.loads((root / "log/00000000.json").read_text())["model"] == "new-model"
