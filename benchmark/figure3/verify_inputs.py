#!/usr/bin/env python3
"""Verify a separately downloaded figure dataset against its frozen manifest."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(data_dir, manifest_path=None):
    root = Path(data_dir).resolve()
    manifest = Path(manifest_path or Path(__file__).with_name('expected_inputs.json'))
    expected = json.loads(manifest.read_text())
    failures = []
    for item in expected['files']:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path escapes data directory')
        if not path.is_file():
            failures.append(item['path'] + ': missing')
        elif path.stat().st_size != item['size_bytes']:
            failures.append(item['path'] + ': size mismatch')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            failures.append(item['path'] + ': checksum mismatch')
    if failures:
        raise ValueError('Figure data verification failed:\n' + '\n'.join(failures))
    return {'verified_files': len(expected['files']), 'source_revision': expected['source_revision']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data)))
