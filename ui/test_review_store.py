"""Review snapshots must never mix sessions or modify source descriptions."""
import json
from pathlib import Path

import pytest

from review_store import frame_reference, review_source, save_review_batch


def source(tmp_path, name, description):
    folder = tmp_path / name
    folder.mkdir()
    (folder / '00000000.json').write_text(json.dumps({'Overall': description, 'Keywords': 'resting'}))
    record = {**frame_reference(folder, 0), 'original': {'Overall': description, 'Keywords': 'resting'},
              'Overall-revised': name + ' revised', 'status': 'Needs correction'}
    return folder, record


def payload(tmp_path, folder, record):
    return {'descriptions': str(folder), 'session_id': review_source(folder)['session_id'],
            'output': str(tmp_path / 'reviews'), 'reviews': {'00000000': record}}


def test_rejects_cross_session_review_without_writing_any_output(tmp_path):
    one, record = source(tmp_path, 'mouse-a', 'Mouse A jumping')
    two, _ = source(tmp_path, 'mouse-b', 'Mouse B resting')
    with pytest.raises(ValueError, match='another session'):
        save_review_batch(payload(tmp_path, two, record))
    assert not (tmp_path / 'reviews').exists()
    assert json.loads((one / '00000000.json').read_text())['Overall'] == 'Mouse A jumping'


def test_rejects_changed_source_and_legacy_unscoped_records(tmp_path):
    folder, record = source(tmp_path, 'mouse-a', 'Original')
    data = payload(tmp_path, folder, record)
    (folder / '00000000.json').write_text('{"Overall":"changed","Keywords":"resting"}')
    with pytest.raises(ValueError, match='Source changed'):
        save_review_batch(data)
    for field in ('session_id', 'source_sha256'):
        current = {**frame_reference(folder, 0), 'original': {'Overall': 'changed'}}
        current.pop(field)
        with pytest.raises(ValueError, match='Source changed'):
            save_review_batch(payload(tmp_path, folder, current))
    assert not (tmp_path / 'reviews').exists()


def test_saves_versioned_copies_and_preserves_originals_and_previous_reviews(tmp_path):
    folder, record = source(tmp_path, 'mouse-a', 'Original')
    data = payload(tmp_path, folder, record)
    first = save_review_batch(data)
    record['Overall-revised'] = 'Second correction'
    second = save_review_batch(data)
    assert first['reviewed_descriptions'] != second['reviewed_descriptions']
    assert json.loads((Path(first['reviewed_descriptions']) / '00000000.json').read_text())['Overall'] == 'mouse-a revised'
    assert json.loads((Path(second['reviewed_descriptions']) / '00000000.json').read_text())['Overall'] == 'Second correction'
    assert json.loads((folder / '00000000.json').read_text())['Overall'] == 'Original'
    assert first['frames'] == 1


def test_rejects_source_destination_and_missing_frames(tmp_path):
    folder, record = source(tmp_path, 'mouse-a', 'Original')
    data = payload(tmp_path, folder, record)
    with pytest.raises(ValueError, match='outside'):
        save_review_batch({**data, 'output': str(folder / 'reviews')})
    with pytest.raises(ValueError, match='exactly one'):
        frame_reference(folder, 12)
    with pytest.raises(ValueError, match='nonnegative'):
        frame_reference(folder, '../file')


def test_rejects_unrelated_original_even_with_current_source_metadata(tmp_path):
    folder, record = source(tmp_path, 'mouse-a', 'Actual source')
    record['original']['Overall'] = 'Unrelated content'
    with pytest.raises(ValueError, match='Original description does not match'):
        save_review_batch(payload(tmp_path, folder, record))
    assert not (tmp_path / 'reviews').exists()


def test_review_session_uses_canonical_directory(tmp_path):
    folder, _ = source(tmp_path, 'mouse-a', 'Original')
    assert review_source(folder) == review_source(folder / '..' / folder.name)
