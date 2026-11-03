"""Local configuration and sample validation, without model or UI dependencies."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mousegpt_config import load_config

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WORKSPACE_DIR = BASE_DIR / "workspace"


def configure_tracing():
    """Tracing is off unless the user explicitly enables it in the environment."""
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGSMITH_TRACING", os.environ["LANGCHAIN_TRACING_V2"])


def azure_openai_enabled():
    """Use Azure OpenAI when its resource endpoint is explicitly configured."""
    return load_config("AGENT", require_key=False).provider == "azure_openai"


def get_api_key(role="AGENT"):
    try:
        return load_config(role).api_key
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def azure_openai_config(deployment_variable, api_version_variable,
                        default_api_version):
    """Return validated Azure resource settings without reading or exposing its key."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise RuntimeError("Set AZURE_OPENAI_ENDPOINT before using Azure OpenAI.")
    deployment = os.environ.get(deployment_variable, "").strip()
    if not deployment:
        raise RuntimeError(f"Set {deployment_variable} before using Azure OpenAI.")
    api_version = os.environ.get(
        api_version_variable,
        os.environ.get("AZURE_OPENAI_API_VERSION", default_api_version),
    ).strip()
    if not api_version:
        raise RuntimeError("AZURE_OPENAI_API_VERSION must not be empty.")
    return endpoint, deployment, api_version


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
