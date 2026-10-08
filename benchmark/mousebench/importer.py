"""Import the recorded NAS audit and archived expert evaluations without inference."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

SAMPLE_RE = re.compile(r'(?<!\d)(\d{6}_C\d+M\d+_\d{8})(?!\d)')
RUBRIC = ('Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords')

def stable_id(prefix, *parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode()
    return prefix + '_' + hashlib.sha256(payload).hexdigest()[:24]

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _json(value, default=None):
    return json.loads(value) if value else ([] if default is None else default)

def _bool(value):
    return value == 'True'

def _model_label(row):
    actual = _json(row.get('actual_model_metadata_json'))
    actual = [x for x in actual if isinstance(x, str)]
    if len(actual) == 1:
        return actual[0]
    path = row['nas_path']
    for marker in ('/output_des/', '/output_img/'):
        if marker in path:
            return path.split(marker)[0].rsplit('/', 1)[-1]
    observed = _json(row.get('observed_model_directories_json'))
    if len(observed) == 1 and isinstance(observed[0], str):
        return observed[0]
    labels = _json(row.get('path_model_labels_json'))
    return labels[0] if len(labels) == 1 and isinstance(labels[0], str) else ''

def import_audit(audit_dir):
    audit_dir = Path(audit_dir)
    frames = []
    with (audit_dir / 'frame_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            frames.append({
                'id': row['sample_id'], 'capture_date': row['capture_date'],
                'group_id': row['group_id'], 'frame_id': row['frame_id'],
                'recording_id': row['recording_id'], 'frame_identity': row['frame_identity'],
                'observed_models': _json(row.get('observed_models_json')),
                'actual_models': _json(row.get('actual_models_json')),
                'evaluation_reference_paths': _json(row.get('evaluation_reference_paths_json')),
                'ground_truth_status': 'no_human_caption_ground_truth_confirmed',
            })
    frame_ids = {f['id'] for f in frames}
    assets = []
    source_hash = file_hash(audit_dir / 'frame_asset_inventory.csv')
    with (audit_dir / 'frame_asset_inventory.csv').open(encoding='utf-8-sig', newline='') as f:
        for line_number, row in enumerate(csv.DictReader(f), 2):
            if row['sample_id'] not in frame_ids:
                raise ValueError('Unknown audit frame: ' + row['sample_id'])
            item = {
                'id': stable_id('asset', row['sample_id'], row['storage_kind'], row['nas_path'], row['archive_member_path'], row['record_locator']),
                'frame_id': row['sample_id'], 'role': row['asset_role'],
                'nas_path': row['nas_path'], 'model_label': _model_label(row),
                'storage_kind': row['storage_kind'], 'relation_level': row['relation_level'],
                'archive_member_path': row['archive_member_path'], 'record_locator': row['record_locator'],
                'size_bytes': int(row['latest_size']) if row.get('latest_size') else None,
                'frame_membership_verified': _bool(row['frame_membership_verified']),
                'original_export_identity_verified': _bool(row['original_export_identity_verified']),
                'evaluation_use_verified': _bool(row['evaluation_use_verified']),
                'is_regular_file': _bool(row['latest_is_regular_file']),
                'is_symlink': _bool(row['latest_is_symlink']),
                'checked_at': row['latest_recorded_checked_at'],
                'actual_models': _json(row.get('actual_model_metadata_json')),
                'observed_model_directories': _json(row.get('observed_model_directories_json')),
                'path_model_labels': _json(row.get('path_model_labels_json')),
                'source_file': 'frame_asset_inventory.csv', 'source_row': line_number,
                'source_file_sha256': source_hash,
            }
            assets.append(item)
    if len({a['id'] for a in assets}) != len(assets):
        raise ValueError('Duplicate audit asset identity')
    return frames, assets

# OOXML cached values are read exactly; formulas are retained, never executed.
_NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
_REL_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
_PAIR_RE = re.compile(r'^(?P<batch>\d{6}-(?:dual-)?production)-(?P<model_a>.+?)##(?P<model_b>.+?)=(?P<sample_id>\d{6}_C\d+M\d+_\d{8})_(?P<language>[a-z]{2})(?:-(?P<reviewer>.*))?$')
_CHOICES = ('A is better', 'B is better', 'Both good', 'Both bad')

def _column(reference):
    return re.sub(r'\d+$', '', reference)

def _column_name(index):
    result = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result

def _cell_value(cell, shared):
    kind = cell.get('t')
    value = cell.find('s:v', _NS)
    text = value.text if value is not None else None
    if kind == 'inlineStr':
        return ''.join(t.text or '' for t in cell.findall('.//s:t', _NS))
    if text is None:
        return None
    if kind == 's':
        return shared[int(text)]
    if kind == 'b':
        return text == '1'
    if kind in ('str', 'e', 'd'):
        return text
    try:
        value = float(text)
        return int(value) if value.is_integer() else value
    except ValueError:
        return text

def read_xlsx(path):
    """Yield (sheet, rows), retaining sparse coordinates and formula caches."""
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            shared = [''.join(t.text or '' for t in si.findall('.//s:t', _NS)) for si in root]
        relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {r.attrib['Id']: r.attrib['Target'] for r in relationships
                   if r.get('TargetMode') != 'External'}
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        for sheet in workbook.findall('s:sheets/s:sheet', _NS):
            target = targets[sheet.attrib['{' + _REL_NS + '}id']]
            target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
            root = ET.fromstring(archive.read(target))
            rows = []
            for row in root.findall('s:sheetData/s:row', _NS):
                cells = {}
                for cell in row.findall('s:c', _NS):
                    formula = cell.find('s:f', _NS)
                    item = {'value': _cell_value(cell, shared)}
                    if formula is not None:
                        item['formula'] = formula.text or ''
                        if formula.attrib:
                            item['formula_attributes'] = dict(formula.attrib)
                    if cell.get('t') == 'e':
                        item['error'] = True
                    if item['value'] is not None or formula is not None:
                        cells[cell.attrib['r']] = item
                if cells:
                    rows.append((int(row.attrib['r']), cells))
            yield sheet.attrib['name'], rows

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = []
        for number, values in enumerate(csv.reader(f), 1):
            cells = {_column_name(i) + str(number): {'value': value} for i, value in enumerate(values, 1) if value != ''}
            if cells:
                rows.append((number, cells))
        yield '(CSV)', rows

def _table(rows):
    for number, cells in rows[:10]:
        headers = {_column(ref): str(cell['value']).strip() for ref, cell in cells.items() if cell['value'] is not None}
        values = set(headers.values())
        if 'image_path' in values and all(key in values for key in RUBRIC):
            return 'rubric', number, headers
        if 'Key' in values and ('Result' in values or 'output' in values):
            return 'pairwise', number, headers
    return None, None, {}

def _raw_row(cells, headers):
    values = {column: item['value'] for ref, item in cells.items() for column in [_column(ref)]}
    raw = {name: values.get(column) for column, name in headers.items()}
    source_cells = {ref: dict(item, header=headers.get(_column(ref), '')) for ref, item in cells.items()}
    return raw, source_cells

def _nonempty(value):
    return value is not None and value != ''

def _observation(kind, raw):
    if kind == 'rubric':
        reference = str(raw.get('image_path') or '')
        match = SAMPLE_RE.search(reference)
        if not match:
            return None, 'missing_or_unparseable_image_path'
        if not any(_nonempty(raw.get(field)) for field in RUBRIC):
            return None, 'source_occurrence_without_rating'
        scores = {field: raw.get(field) for field in RUBRIC}
        basename = reference.replace('\\', '/').rsplit('/', 1)[-1]
        path_model = basename.split(match.group(1), 1)[0].rstrip('_')
        cell_model = str(raw.get('model') or '').strip()
        issues = []
        if cell_model and cell_model != path_model:
            issues.append('model_cell_disagrees_with_image_path')
        invalid = [key for key, value in scores.items() if _nonempty(value) and value not in (0, 1, '0', '1')]
        if invalid:
            issues.append('nonbinary_rubric_values:' + ','.join(invalid))
        # Preserve numeric cells as read; missing fields remain null, not zero.
        return {'frame_id': match.group(1), 'kind': kind, 'model_label': cell_model or path_model,
                'path_model_label': path_model, 'value': scores,
                'reviewer': str(raw.get('usr') or ''), 'image_path': reference,
                'missing_rubric_fields': [key for key, value in scores.items() if not _nonempty(value)],
                'issues': issues}, None
    key = str(raw.get('Key') or '')
    match = _PAIR_RE.fullmatch(key)
    if not match:
        return None, 'missing_or_unparseable_pair_key'
    parsed = match.groupdict()
    result = str(raw.get('Result') or raw.get('output') or '')
    value, timestamp = None, ''
    for choice in _CHOICES:
        preference_match = re.fullmatch(re.escape(choice) + r'\s*(?:-\s*(.*))?', result.strip())
        if preference_match:
            value = choice
            timestamp = preference_match.group(1) or ''
            break
    cached = str(raw.get('Results') or '').strip()
    issues = []
    if value is None and cached in _CHOICES:
        value = cached
        issues.append('preference_from_cached_results_only')
    if value is None:
        return None, 'source_occurrence_without_valid_preference'
    if cached and cached != value:
        issues.append('cached_results_disagree_with_raw_result')
    model_a = str(raw.get('ModelA') or parsed['model_a'])
    model_b = str(raw.get('ModelB') or parsed['model_b'])
    if model_a != parsed['model_a'] or model_b != parsed['model_b']:
        issues.append('model_cells_disagree_with_pair_key')
    cell_time = str(raw.get('Time') or '').strip()
    if cell_time and timestamp and cell_time != timestamp:
        issues.append('time_cell_disagrees_with_result_timestamp')
    if (cell_time or timestamp) and not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?', cell_time or timestamp):
        issues.append('unknown_or_invalid_timestamp')
    reviewer = str(raw.get('Username') or parsed['reviewer'] or '')
    if any(token in reviewer for token in ('##', '=', '/')):
        issues.append('invalid_derived_reviewer_cell')
        reviewer = parsed['reviewer'] or ''
    if not reviewer:
        issues.append('reviewer_unknown')
    if raw.get('Username') and parsed['reviewer'] and reviewer != parsed['reviewer']:
        issues.append('reviewer_cell_disagrees_with_pair_key')
    return {'frame_id': parsed['sample_id'], 'kind': kind, 'value': value,
            'model_a': model_a, 'model_b': model_b,
            'key_model_a': parsed['model_a'], 'key_model_b': parsed['model_b'],
            'reviewer': reviewer, 'timestamp': cell_time or timestamp,
            'batch': parsed['batch'], 'language': parsed['language'], 'comparison_key': key,
            'issues': issues}, None

def import_ratings(evaluation_dir, frame_ids):
    evaluation_dir = Path(evaluation_dir)
    ratings, sources, rejected = [], [], []
    for path in sorted(evaluation_dir.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in ('.xlsx', '.csv') or path.name.startswith('~$'):
            continue
        relative = path.relative_to(evaluation_dir).as_posix()
        digest = file_hash(path)
        source = {'path': relative, 'sha256': digest, 'size_bytes': path.stat().st_size,
                  'sheets': [], 'rating_count': 0}
        reader = read_xlsx if path.suffix.lower() == '.xlsx' else read_csv
        for sheet, rows in reader(path):
            kind, header_row, headers = _table(rows)
            sheet_info = {'sheet': sheet, 'kind': kind or 'derived_or_other', 'rating_count': 0,
                          'nonempty_rows': len(rows)}
            source['sheets'].append(sheet_info)
            if kind is None:
                sheet_info['status'] = 'not_imported_as_expert_observations'
                continue
            sheet_info['header_row'] = header_row
            sheet_info['headers'] = headers
            for row_number, cells in rows:
                if row_number <= header_row:
                    continue
                raw, source_cells = _raw_row(cells, headers)
                observation, reason = _observation(kind, raw)
                if observation and observation['frame_id'] not in frame_ids:
                    observation, reason = None, 'frame_not_in_audit'
                if observation is None:
                    # Ignore formula-fill rows whose identifying reference is absent.
                    reference = raw.get('image_path') if kind == 'rubric' else raw.get('Key')
                    if _nonempty(reference):
                        rejected.append({'source_file': relative, 'source_file_sha256': digest,
                                         'sheet': sheet, 'row': row_number, 'reason': reason,
                                         'raw': raw, 'source_cells': source_cells})
                    continue
                observation.update({
                    'id': stable_id('rating', relative, digest, sheet, row_number, kind),
                    'source_file': relative, 'source_file_sha256': digest,
                    'sheet': sheet, 'row': row_number, 'raw': raw, 'source_cells': source_cells,
                    'linkage_status': 'frame_only_unverified_output',
                    'candidate_id': None, 'annotation_type': 'archived_expert_evaluation',
                    'is_ground_truth_caption': False,
                })
                fingerprint_parts = [kind, observation['frame_id'], observation.get('model_label'),
                                     observation.get('model_a'), observation.get('model_b'),
                                     observation['reviewer'], observation.get('timestamp'), observation['value']]
                observation['duplicate_fingerprint'] = stable_id('observation', *fingerprint_parts)
                ratings.append(observation)
                source['rating_count'] += 1
                sheet_info['rating_count'] += 1
        sources.append(source)
    duplicate_counts = collections.Counter(r['duplicate_fingerprint'] for r in ratings)
    for rating in ratings:
        rating['matching_observation_copies'] = duplicate_counts[rating['duplicate_fingerprint']]
    return ratings, sources, rejected

def import_catalog(audit_dir, evaluation_dir):
    """Build a source-backed catalog; repeated versions remain distinct evidence."""
    frames, assets = import_audit(audit_dir)
    ratings, sources, rejected = import_ratings(evaluation_dir, {frame['id'] for frame in frames})
    kinds = collections.Counter(r['kind'] for r in ratings)
    groups = collections.Counter(r['duplicate_fingerprint'] for r in ratings)
    summary = {
        'schema_version': 1, 'frame_count': len(frames), 'asset_count': len(assets),
        'rating_count': len(ratings), 'ratings_count': len(ratings),
        'ratings_by_kind': dict(kinds), 'ratings_import_status': 'complete',
        'frames_with_ratings': len({r['frame_id'] for r in ratings}),
        'evaluation_sources': sources, 'evaluation_file_count': len(sources),
        'audit_sources': [{'path': name, 'sha256': file_hash(Path(audit_dir) / name)}
                          for name in ('frame_summary.csv', 'frame_asset_inventory.csv')],
        'source_occurrences_without_imported_rating': rejected,
        'excluded_observation_count': len(rejected),
        'identical_observation_fingerprints': len(groups),
        'repeated_observation_groups': sum(count > 1 for count in groups.values()),
        'rows_in_repeated_observation_groups': sum(count for count in groups.values() if count > 1),
        'ratings_with_issues': sum(bool(r['issues']) for r in ratings),
        'asset_roles': dict(collections.Counter(a['role'] for a in assets)),
        'limitations': [
            'Ratings retain archived source versions, original/output sheets and repeated copies; row totals are not independent expert observations.',
            'Duplicate fingerprints identify identical imported payloads, not adjudicated independent judgments; differing corrections are retained.',
            'Ratings link by frame and source model label only; the exact evaluated candidate output is unverified.',
            'Human rubric and pairwise judgments are evidence, not human-written caption ground truth.',
            'Spreadsheet formulas use saved cached values; formulas are preserved without recalculation.',
            'Recording-level containers do not establish per-frame contents; NAS paths reflect the recorded audit, not a new live scan.',
        ],
    }
    return {'frames': frames, 'assets': assets, 'ratings': ratings, 'summary': summary}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', required=True, type=Path)
    parser.add_argument('--evaluation-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    catalog = import_catalog(args.audit_dir, args.evaluation_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(json.dumps({key: catalog['summary'][key] for key in ('frame_count', 'asset_count', 'rating_count', 'ratings_by_kind', 'excluded_observation_count')}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
