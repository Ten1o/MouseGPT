"""The workflow entry point is local-only and does not need model credentials."""
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_workflow", ROOT / "scripts/check_workflow.py")
guide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guide)


def test_descriptions_take_priority_over_existing_stage2_inputs(tmp_path):
    (tmp_path / "descriptions").mkdir()
    (tmp_path / "descriptions/00000001.json").write_text("{}")
    (tmp_path / "pre_images/data3").mkdir(parents=True)
    (tmp_path / "pre_images/data5").mkdir(parents=True)
    (tmp_path / "pre_images/data3/00000001.jpg").touch()
    (tmp_path / "pre_images/data5/00000001.jpg").touch()
    (tmp_path / "velocity.json").write_text("{}")
    assert guide.inspect_input(tmp_path)[0] == "3"


def test_stage2_partial_inputs_are_reported_without_model_preflight(tmp_path):
    (tmp_path / "pre_images/data3").mkdir(parents=True)
    (tmp_path / "pre_images/data3/00000001.jpg").touch()
    stage, _, notes, role = guide.inspect_input(tmp_path)
    assert stage == "2" and role is None
    assert any("Camera 3/5" in note for note in notes)
    assert any("velocity.json" in note for note in notes)


def test_complete_stage2_inputs_select_vision_role(tmp_path):
    for camera in (3, 5):
        folder = tmp_path / "pre_images" / f"data{camera}"
        folder.mkdir(parents=True)
        (folder / "00000001.jpg").touch()
    (tmp_path / "velocity.json").write_text("{}")
    stage, _, _, role = guide.inspect_input(tmp_path)
    assert (stage, role) == ("2", "VISION")


def test_multiview_recording_selects_stage1_without_api(tmp_path):
    videos = tmp_path / "videos"
    calibration = tmp_path / "calibration"
    videos.mkdir()
    calibration.mkdir()
    for camera in range(1, 9):
        (videos / f"data{camera}_short.mp4").touch()
        (calibration / f"Camera{camera}_params.mat").touch()
    stage, _, notes, role = guide.inspect_input(tmp_path)
    assert (stage, role) == ("1", None)
    assert any("8/8" in note for note in notes)


def test_hdf5_requires_matching_previews_for_stage4(tmp_path):
    h5 = tmp_path / "session.h5"
    h5.touch()
    assert guide.inspect_input(h5)[0] == "3"
    previews = tmp_path / "session_imgs"
    previews.mkdir()
    (previews / "00000001.jpg").touch()
    assert guide.inspect_input(h5)[0] == "4"


def test_model_status_hides_key_and_never_contacts_provider():
    with patch.dict(os.environ, {}, clear=True), patch(
            "mousegpt_config.read_env_file", return_value={
                "MOUSEGPT_API_KEY": "very-secret-test-key"}):
        status = guide.model_status("VISION")
    assert "VISION configuration present" in status
    assert "very-secret-test-key" not in status
