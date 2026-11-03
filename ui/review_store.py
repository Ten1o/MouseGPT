"""Session-bound review snapshots. This module needs only the standard library."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mousegpt_descriptions import read_description


def review_source(directory):
    if not isinstance(directory, (str, Path)) or not str(directory).strip():
        raise ValueError('Select a description directory before reviewing frames.')
    source = Path(directory).expanduser().resolve()
    if not source.is_dir():
        raise ValueError('Description directory does not exist.')
    identity = hashlib.sha256(os.path.normcase(str(source)).encode()).hexdigest()
    return {'session_id': identity, 'description_directory': str(source)}


def frame_reference(directory, frame):
    context = review_source(directory)
    value = str(frame)
    if not value.isascii() or not value.isdigit():
        raise ValueError('Review frame ID must be a nonnegative integer.')
    identifier = str(int(value)).zfill(8)
    candidates = [p for p in Path(context['description_directory']).iterdir()
                  if p.is_file() and p.suffix.lower() in {'.json', '.txt'}
                  and p.stem.isascii() and p.stem.isdigit()
                  and str(int(p.stem)).zfill(8) == identifier]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one source description for frame {identifier}.')
    file = candidates[0]
    if file.resolve().parent != Path(context['description_directory']):
        raise ValueError('A review source file must belong to its description directory.')
    return {**context, 'frame_id': identifier, 'source_file': file.name,
            'source_sha256': hashlib.sha256(file.read_bytes()).hexdigest()}


def save_review_batch(body):
    context = review_source(body.get('descriptions'))
    if body.get('session_id') != context['session_id']:
        raise ValueError('Review session changed. Reload its frames before saving.')
    records = body.get('reviews')
    if not isinstance(records, dict) or not records:
        raise ValueError('Load and review at least one frame before saving.')
    if not isinstance(body.get('output'), str) or not body['output'].strip():
        raise ValueError('Select an output directory for the review snapshot.')
    dest = Path(body['output']).expanduser().resolve()
    source = Path(context['description_directory'])
    if dest == source or source in dest.parents:
        raise ValueError('Save reviewed copies outside the source description directory.')
    prepared = {}
    for frame, record in records.items():
        if not isinstance(record, dict):
            raise ValueError('Invalid review record.')
        expected = frame_reference(source, frame)
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Source changed or belongs to another session: frame {frame}. Reload before saving.')
        original = record.get('original')
        if not isinstance(original, dict) or not original or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in original.items()):
            raise ValueError('Visual review requires a structured description with text fields.')
        actual = read_description(source / expected['source_file'], expected_sha256=expected['source_sha256'])
        if original != actual:
            raise ValueError(f'Original description does not match source frame {frame}. Reload before saving.')
        updated = {}
        for key, value in original.items():
            revision = record.get(key + '-revised', '')
            if not isinstance(revision, str):
                raise ValueError('Revised description fields must contain text.')
            updated[key] = revision.strip() or value
        if expected['frame_id'] in prepared:
            raise ValueError('Duplicate review frame ID.')
        prepared[expected['frame_id']] = updated
    # Validate every record before creating any output. Each save is a new snapshot.
    dest.mkdir(parents=True, exist_ok=True)
    name = 'review-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(6)
    target = dest / name
    staging = Path(tempfile.mkdtemp(prefix='.review-', dir=dest))
    try:
        copies = staging / 'descriptions'
        copies.mkdir()
        for frame, description in prepared.items():
            (copies / f'{frame}.json').write_text(json.dumps(description, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        snapshot = {'schemaVersion': 3, **context, 'reviews': records}
        (staging / 'reviews.json').write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {'path': str(target / 'reviews.json'), 'reviewed_descriptions': str(target / 'descriptions'),
            'session_id': context['session_id'], 'frames': len(prepared)}
