"""Batch launchers preserve configured paths as subprocess arguments."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import Mock

import pytest

DEMO = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured_paths(tmp_path, monkeypatch):
    root = tmp_path / 'input data; literal shell characters'
    videos = root / '20241009' / 'session' / 'videos'
    videos.mkdir(parents=True)
    for name in ('view 1.mp4', 'view 2.mp4'):
        (videos / name).touch()
    settings = {
        'MOUSEGPT_DATA_ROOT': str(root),
        'MOUSEGPT_DATES': '20241009',
        'MOUSEGPT_GROUP_GLOB': 'session',
        'MOUSEGPT_VIDEO_DIR': str(videos),
        'MOUSEGPT_VIDEO_GLOB': str(videos / '*.mp4'),
        'MOUSEGPT_OUTPUT_DIR': str(tmp_path / 'output data; literal'),
        'MOUSEGPT_POSE_CONFIG': str(tmp_path / 'pose config.py'),
        'MOUSEGPT_POSE_WEIGHTS': str(tmp_path / 'pose checkpoint.pth'),
        'MOUSEGPT_DETECTOR_CONFIG': str(tmp_path / 'detector config.py'),
        'MOUSEGPT_DETECTOR_WEIGHTS': str(tmp_path / 'detector checkpoint.pth'),
    }
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    return settings, videos


@pytest.mark.parametrize('script, expected_count, selected_video', [
    ('batch_infer.py', 2, 'view 1.mp4'),
    ('analyze_top_view/batch_infer_topview.py', 2, 'view 1.mp4'),
    ('run_infer.py', 1, 'view 2.mp4'),
    ('run_infer_2mouse_rich.py', 1, 'view 1.mp4'),
])
def test_batch_preserves_paths_and_existing_video_selection(script, expected_count, selected_video, configured_paths, monkeypatch):
    settings, videos = configured_paths
    launch = Mock()
    monkeypatch.setattr(subprocess, 'run', launch)
    runpy.run_path(str(DEMO / script), run_name='__main__')
    assert launch.call_count == expected_count
    command = launch.call_args_list[0].args[0]
    assert isinstance(command, list)
    assert command[0] == sys.executable
    assert Path(command[1]).is_file()
    assert command[2] == str(videos / selected_video)
    for option, variable in [('--pose2d', 'MOUSEGPT_POSE_CONFIG'), ('--pose2d-weights', 'MOUSEGPT_POSE_WEIGHTS'), ('--det-model', 'MOUSEGPT_DETECTOR_CONFIG'), ('--det-weights', 'MOUSEGPT_DETECTOR_WEIGHTS')]:
        assert command[command.index(option) + 1] == settings[variable]
    assert launch.call_args_list[0].kwargs == {'check': True}


def test_failed_inference_stops_batch(configured_paths, monkeypatch):
    launch = Mock(side_effect=subprocess.CalledProcessError(1, ['offline-test']))
    monkeypatch.setattr(subprocess, 'run', launch)
    with pytest.raises(subprocess.CalledProcessError):
        runpy.run_path(str(DEMO / 'batch_infer.py'), run_name='__main__')
    assert launch.call_count == 1


def test_unconfigured_input_fails_before_inference(monkeypatch):
    monkeypatch.delenv('MOUSEGPT_VIDEO_DIR', raising=False)
    launch = Mock()
    monkeypatch.setattr(subprocess, 'run', launch)
    with pytest.raises(KeyError, match='MOUSEGPT_VIDEO_DIR'):
        runpy.run_path(str(DEMO / 'batch_infer.py'), run_name='__main__')
    launch.assert_not_called()


def test_annotation_launcher_uses_checked_arguments(tmp_path, monkeypatch):
    labels = tmp_path / 'calibration files'
    labels.mkdir()
    calibration = labels / '241009_G1_sample.mat'
    calibration.touch()
    monkeypatch.setenv('MOUSEGPT_CALIBRATION_DIR', str(labels))
    monkeypatch.setenv('MOUSEGPT_DATA_ROOT', str(tmp_path / 'raw data'))
    monkeypatch.setenv('MOUSEGPT_OUTPUT_DIR', str(tmp_path / 'annotation output'))
    launch = Mock()
    monkeypatch.setattr(subprocess, 'run', launch)
    runpy.run_path(str(DEMO / 'check/batch_vis_write.py'), run_name='__main__')
    command = launch.call_args.args[0]
    assert command[command.index('--label_path') + 1] == str(calibration)
    assert Path(command[1]).is_file()
    assert launch.call_args.kwargs == {'check': True}


def test_notebook_configuration_reports_missing_inputs(monkeypatch):
    notebook = json.loads((DEMO / 'velocity/pos_vol.ipynb').read_text())
    setup = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
    monkeypatch.delenv('MOUSEGPT_KEYPOINTS_NPY', raising=False)
    with pytest.raises(ValueError, match='MOUSEGPT_KEYPOINTS_NPY'):
        exec(compile(''.join(setup['source']), '<notebook configuration>', 'exec'), {})
