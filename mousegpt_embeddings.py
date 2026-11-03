"""Embedding provenance shared by session packaging and semantic queries.

Vector length alone does not identify an embedding space. Sessions retain the
exact configured service, model/deployment and request settings. No credential
is included in these records.
"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
# The immutable nlui-demo HDF5 distributed in assets/manifest.json predates
# provenance attributes. Its documented model is text-embedding-3-large.
# Recognize these exact bytes only; a filename or caller-supplied receipt is
# insufficient evidence for an arbitrary legacy session.
LEGACY_DEMO_SHA256 = "cb49ccbdebcc051c4d48f682b5998f521c4bd905a61c928edd1f135a5a32f765"
LEGACY_DEMO_BYTES = 1198045200
LEGACY_DEMO_IDENTITY = {
    "schema_version": SCHEMA_VERSION,
    "provider": "openai",
    "endpoint": "https://api.openai.com/v1",
    "model": "text-embedding-3-large",
    "api_version": None,
    "settings": {},
    "dimensions": 3072,
}


def provider_identity(config):
    """Return every configurable embedding request property except its key."""
    return {
        "provider": config.provider,
        "endpoint": config.endpoint,
        "model": config.model,
        "api_version": config.api_version,
        # No optional dimensions/task/normalization settings are currently sent.
        "settings": {},
    }


def embedding_identity(config, dimensions):
    return validate_identity({
        "schema_version": SCHEMA_VERSION,
        **provider_identity(config),
        "dimensions": dimensions,
    })


def validate_identity(value):
    """Reject incomplete metadata instead of assuming a historical default."""
    required = {"schema_version", "provider", "endpoint", "model",
                "api_version", "settings", "dimensions"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Missing or invalid embedding provenance; regenerate embeddings and repackage the session.")
    if (type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION
            or type(value["dimensions"]) is not int or value["dimensions"] < 1
            or not all(isinstance(value[field], str) and value[field].strip()
                       for field in ("provider", "endpoint", "model"))
            or value["api_version"] is not None and not isinstance(value["api_version"], str)
            or not isinstance(value["settings"], dict)):
        raise ValueError("Invalid embedding provenance; regenerate embeddings and repackage the session.")
    return dict(value)


def request_digest(api_identity, content):
    encoded = json.dumps(
        {"api": api_identity, "input": content, "model": api_identity["model"]},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _known_legacy_file(path, device, inode, size, modified_ns, changed_ns):
    # Stat values are part of the cache key: replacement or modification must
    # trigger a new checksum. Unknown small files do not require hashing.
    return size == LEGACY_DEMO_BYTES and file_digest(path) == LEGACY_DEMO_SHA256


def read_session_embedding_identity(path, store):
    """Read HDF5 provenance or recognize the exact published legacy sample."""
    raw = getattr(store.get_storer("features_des").attrs, "embedding_identity", None)
    if raw is not None:
        try:
            return validate_identity(json.loads(raw))
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid session embedding provenance; regenerate embeddings and repackage the session.") from error
    path = Path(path).resolve()
    stat = path.stat()
    if _known_legacy_file(str(path), stat.st_dev, stat.st_ino, stat.st_size,
                          stat.st_mtime_ns, stat.st_ctime_ns):
        return dict(LEGACY_DEMO_IDENTITY)
    raise ValueError(
        "This session has no verified embedding provenance. Regenerate its descriptions' "
        "embeddings with 3_applications/embedding_request.py and rebuild it with "
        "3_applications/create_h5.py. Only the exact published nlui-demo is recognized automatically.")


def require_compatible_embeddings(path, store, config, dimensions):
    stored = read_session_embedding_identity(path, store)
    current = embedding_identity(config, dimensions)
    differences = sorted(key for key in current if current[key] != stored[key])
    if differences:
        raise ValueError(
            "Embedding configuration does not match the session (" + ", ".join(differences) + "). "
            "Use the original EMBEDDING configuration or regenerate all session embeddings and repackage it.")
    return stored
