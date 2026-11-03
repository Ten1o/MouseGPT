"""Bind environment checks to the application and dependency contracts tested."""
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def application_fingerprint(profile, root=ROOT):
    root = Path(root)
    paths = {"ui/environment_probe.py", "ui/environment_validation.py"}
    if profile == "agent":
        paths.update({"requirements-api.txt", "requirements-assets.txt",
                      "4_NLUI/requirements.txt", "mousegpt_config.py",
                      "mousegpt_embeddings.py"})
        for pattern in ("4_NLUI/*.py", "4_NLUI/tools_function/*.py"):
            paths.update(str(path.relative_to(root)) for path in root.glob(pattern))
    elif profile == "core":
        paths.update({"ui/environments/core.in", "requirements-api.txt",
                      "ui/worker.py", "mousegpt_config.py", "mousegpt_embeddings.py",
                      "1_keypoints/demo/batch_vis.py", "1_keypoints/demo/video_utils.py"})
    elif profile == "pose":
        paths.update({"1_keypoints/demo/inferencer_demo.py",
                      "1_keypoints/demo/triangulation.py"})
    else:
        raise ValueError("Unknown environment profile.")
    digest = hashlib.sha256(profile.encode())
    for relative in sorted(paths):
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validation_is_current(profile, report):
    return bool(report.get("passed") and
                report.get("application_fingerprint") == application_fingerprint(profile))
