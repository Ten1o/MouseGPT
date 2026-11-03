"""Inspect local MouseGPT inputs and API settings without running a pipeline."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mousegpt_config import load_config


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DESCRIPTION_SUFFIXES = {".json", ".txt"}


def _stems(folder, suffixes):
    if not folder.is_dir():
        return set()
    return {path.stem for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes}


def inspect_input(path):
    """Return a stage suggestion and diagnostics based only on local filenames."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.suffix.lower() != ".h5":
            raise ValueError("Provide a session directory or an HDF5 session file.")
        previews = path.with_name(path.stem + "_imgs")
        if _stems(previews, IMAGE_SUFFIXES):
            return "4", path, ["HDF5 and matching preview directory found.",
                                "The current interface lists sessions under 4_NLUI/data; place this pair there before launch if stored elsewhere."], "AGENT"
        return "3", path, ["HDF5 found, but matching *_imgs previews are missing; Stage 4 cannot display this session yet."], None

    h5_files = sorted(path.glob("*.h5"))
    ready_h5 = [file for file in h5_files if _stems(file.with_name(file.stem + "_imgs"), IMAGE_SUFFIXES)]
    if len(ready_h5) == 1:
        return "4", ready_h5[0], ["HDF5 and matching preview directory found.",
                                 "The current interface lists sessions under 4_NLUI/data; place this pair there before launch if stored elsewhere."], "AGENT"
    if len(ready_h5) > 1:
        raise ValueError("Multiple Stage 4 sessions found; provide one .h5 path.")

    descriptions = _stems(path / "descriptions", DESCRIPTION_SUFFIXES)
    if descriptions:
        features = _stems(path / "feature", {".csv"})
        notes = [f"Found {len(descriptions)} frame description(s)."]
        if features:
            notes.append("Feature files exist; the embedding script validates its cache and fills missing entries.")
        return "3", path, notes, "EMBEDDING"

    camera3 = _stems(path / "pre_images" / "data3", IMAGE_SUFFIXES)
    camera5 = _stems(path / "pre_images" / "data5", IMAGE_SUFFIXES)
    if camera3 or camera5 or (path / "velocity.json").exists():
        notes = []
        if not camera3 or camera3 != camera5:
            notes.append("Camera 3/5 image frame IDs must be the same nonempty set.")
        if not (path / "velocity.json").is_file():
            notes.append("Missing velocity.json.")
        if notes:
            return "2", path, notes, None
        return "2", path, [f"Found {len(camera3)} paired image frame(s) and velocity.json."], "VISION"

    video_dir = path / "videos" if (path / "videos").is_dir() else path
    calibration = path / "calibration"
    videos = {number for number in range(1, 9)
              if any(video_dir.glob(f"data{number}*.mp4"))}
    cameras = {number for number in range(1, 9)
               if (calibration / f"Camera{number}_params.mat").is_file()}
    if videos or cameras:
        notes = [f"Found {len(videos)}/8 camera videos and {len(cameras)}/8 calibration files."]
        notes.append("Stage 1 also requires compatible detector and 24-keypoint pose configurations and weights.")
        return "1", path, notes, None

    raise ValueError("No recognized session inputs. See CONFIGURATION.md for supported layouts.")


def model_status(role):
    """Validate role settings without exposing the key or contacting a service."""
    if role is None:
        return "No model API is needed for this local check."
    try:
        config = load_config(role)
    except ValueError as error:
        return f"{role} configuration needs attention: {error}"
    return (f"{role} configuration present: {config.provider}, "
            f"model {config.model}. Key value is hidden; access was not tested.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Session directory or one session .h5 file")
    args = parser.parse_args(argv)
    try:
        stage, selected, notes, role = inspect_input(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Suggested starting stage: {stage}")
    print(f"Input: {selected}")
    for note in notes:
        print(f"- {note}")
    print(model_status(role))
    print(f"Guide: {ROOT / {'1': '1_keypoints', '2': '2_VLM', '3': '3_applications', '4': '4_NLUI'}[stage] / 'readme.md'}")
    print("No files were changed and no API request was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
