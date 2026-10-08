"""Local configuration and sample validation, without model or UI dependencies."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WORKSPACE_DIR = BASE_DIR / "workspace"


def configure_tracing():
    """Tracing is off unless the user explicitly enables it in the environment."""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGSMITH_TRACING", os.environ["LANGCHAIN_TRACING_V2"])


def get_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("YOUR_"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment before starting the NLUI.")
    return key


def image_files(config):
    images = sorted(Path(config["img_folder"]).glob("*.jpg"))
    if "frame_ids" in config:
        allowed = set(config["frame_ids"])
        images = [image for image in images if image.stem in allowed]
    return images


def session_config(name, data_dir=DATA_DIR):
    """Validate an existing paired HDF5/image session without opening its HDF5."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Select a session name from the downloaded sample data.")
    data_dir = Path(data_dir)
    config = {
        "file_path": str(data_dir / f"{name}.h5"),
        "img_folder": str(data_dir / f"{name}_imgs"),
        "workspace": str(WORKSPACE_DIR / name),
    }
    if not Path(config["file_path"]).is_file() or not image_files(config):
        raise FileNotFoundError(
            f"Session {name} needs both its HDF5 file and JPEG image folder in {data_dir}. "
            "Download the NLUI sample with the repository's sample-data downloader first."
        )
    return config


def available_sessions(data_dir=DATA_DIR):
    sessions = []
    for file in sorted(Path(data_dir).glob("*.h5")):
        try:
            session_config(file.stem, data_dir)
        except FileNotFoundError:
            continue
        sessions.append(file.stem)
    return sessions


def validate_frame_ids(description_ids, feature_ids):
    """Prevent positional matching from silently pairing different frame rows."""
    descriptions = [str(value).zfill(8) for value in description_ids]
    features = [str(value).zfill(8) for value in feature_ids]
    if not descriptions or len(set(descriptions)) != len(descriptions):
        raise ValueError("Description frame IDs must be nonempty and unique.")
    if descriptions != features:
        raise ValueError("Description and feature frame IDs must match in the same row order.")
    return descriptions
