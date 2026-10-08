"""Extract camera 3 and 5 images while retaining original frame indices."""

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


FRAME_STEP = 12


def rename(raw_fps, output_folder_temp, output_folder, start_time):
    """Copy numbered FFmpeg outputs to frame-index filenames without deleting inputs."""
    source = Path(output_folder_temp)
    destination = Path(output_folder)
    destination.mkdir(parents=True, exist_ok=True)
    start_frame = int(raw_fps * 60 * start_time)
    images = sorted(source.glob("*.jpg"))
    if not images:
        raise ValueError(f"No frames were extracted into {source}")
    for path in images:
        index = int(path.stem) - 1
        if index < 0:
            raise ValueError(f"Expected FFmpeg numbering to start at 1: {path.name}")
        target = destination / f"{start_frame + index * FRAME_STEP:08d}.jpg"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        shutil.copy2(path, target)


def extract_frames(input_folder, output_folder, start_time, end_time, fps, ffmpeg="ffmpeg"):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and greater than zero")
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError("Start and end times must be finite")
    if start_time < 0 or (end_time != 0 and end_time <= start_time):
        raise ValueError("Times must satisfy 0 <= start_time < end_time; end_time=0 means EOF")
    start_frame = int(fps * 60 * start_time)
    end_frame = int(fps * 60 * end_time) if end_time else None
    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("The selected time interval contains no frames")
    trim = f"trim=start_frame={start_frame}"
    if end_frame is not None:
        trim += f":end_frame={end_frame}"
    # The select counter starts at zero after trim, matching the filename offset.
    filters = trim + rf",select=not(mod(n\,{FRAME_STEP})),setpts=PTS-STARTPTS"
    source = Path(input_folder)
    images_folder = Path(output_folder) / "images"
    jobs = [(source / f"data{camera}.mp4", images_folder / f"data{camera}")
            for camera in (3, 5)]
    for input_file, destination in jobs:
        if not input_file.is_file():
            raise FileNotFoundError(input_file)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}; choose a new output folder")
    images_folder.mkdir(parents=True, exist_ok=True)
    # Publish only after both videos succeed. Only this call's temporary tree is removed.
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=images_folder) as temporary:
        temporary = Path(temporary)
        staged = []
        for input_file, destination in jobs:
            raw = temporary / f"{destination.name}-raw"
            raw.mkdir()
            ready = temporary / destination.name
            command = [str(ffmpeg), "-nostdin", "-n", "-i", str(input_file),
                       "-filter:v", filters, "-vsync", "0", str(raw / "%08d.jpg")]
            subprocess.run(command, check=True)
            rename(fps, raw, ready, start_time)
            staged.append((ready, destination))
        for ready, destination in staged:
            if destination.exists():
                raise FileExistsError(f"Output appeared during extraction: {destination}")
            ready.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_folder", required=True, help="Folder containing data3.mp4 and data5.mp4")
    parser.add_argument("--output_folder", required=True, help="New output folder for extracted images")
    parser.add_argument("--start_time", type=int, required=True, help="Start time in minutes")
    parser.add_argument("--end_time", type=int, default=20, help="End time in minutes; 0 means EOF")
    parser.add_argument("--fps", type=float, required=True, help="Original video frame rate")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable path")
    args = parser.parse_args()
    extract_frames(args.input_folder, args.output_folder, args.start_time, args.end_time, args.fps, args.ffmpeg)


if __name__ == "__main__":
    main()
