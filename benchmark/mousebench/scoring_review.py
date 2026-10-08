"""Read-only inspection of immutable judgments and their exact evidence versions.

Current eligibility deliberately follows Store.scorecards. Historical evidence
can still be reviewed after the current reference/input changes; it is never
silently replaced with the newest reference or image.
"""
import hashlib
import json
import re
from collections import Counter
from urllib.parse import quote

from .evaluation import analyze_output, score_judgment
from .protocol import benchmark_index
from .store import digest

SCHEMA = 'mousebench.scoring-review.v1'
_SCORE_FIELDS = ('schema_version', 'frame_id', 'run_id', 'protocol_version', 'judge',
    'rubric_sha256', 'ground_truth_sha256', 'prediction_sha256', 'input_sha256',
    'claims', 'supported_fact_ids', 'fact_coverage', 'structure_score',
    'instruction_score', 'notes', 'track', 'recording_id')


def _rubrics(store):
    if not store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='wf_records'").fetchone():
        return {}
    return {r['sha256']: r for row in store.db.execute("SELECT data FROM wf_records WHERE kind='rubric'")
            for r in [json.loads(row[0])]}


def _judgments(store, run_id):
    return [json.loads(row[0]) for row in store.db.execute(
        'SELECT data FROM judgments WHERE run_id=? ORDER BY frame_id,id', (run_id,))]


def _judgment_headers(store):
    """Read group identity without materializing every assessment's claims/logs."""
    rows = store.db.execute("""SELECT id,run_id,frame_id,
        json_extract(data,'$.schema_version') AS schema_version,
        json_extract(data,'$.protocol_version') AS protocol_version,
        json_extract(data,'$.rubric_sha256') AS rubric_sha256,
        json_extract(data,'$.judge') AS judge,
        json_extract(data,'$.track') AS track,
        json_extract(data,'$.input_track') AS input_track,
        json_extract(data,'$.scores.scoring_version') AS scoring_version
        FROM judgments ORDER BY frame_id,id""")
    result = []
    for row in rows:
        entry = dict(row)
        entry['judge'] = json.loads(entry['judge'])
        entry['scores'] = {'scoring_version': entry.pop('scoring_version')}
        result.append(entry)
    return result


def _versions(store, kind, frame_id, context=None):
    if context is None:
        return store.versions(kind, frame_id)
    key = (kind, frame_id)
    if key not in context['versions']:
        context['versions'][key] = store.versions(kind, frame_id)
    return context['versions'][key]


def _group(j, rubric=None):
    scoring = j.get('scores', {}).get('scoring_version') or (
        'mousebench.graded-factual-f1.v2' if j.get('schema_version') == 'mousebench.judgment.v2'
        else 'mousebench.factual-f1.v1')
    key = digest([j['protocol_version'], j['rubric_sha256'], j['judge'], j['track'], j.get('input_track'), scoring])
    return {'group_id': key, 'judge': j['judge'], 'rubric_sha256': j['rubric_sha256'],
            'rubric_id': (rubric or {}).get('id'), 'rubric_name': (rubric or {}).get('content', {}).get('id'),
            'rubric_status': (rubric or {}).get('status'), 'protocol_version': j['protocol_version'],
            'scoring_version': scoring, 'track': j['track'], 'input_track': j.get('input_track')}


def _reason(code, message):
    return {'code': code, 'message': message}


def _eligibility(store, j, context=None):
    """The reference/input and critical-issue gates used by Store.scorecards."""
    if context is None:
        gt = store.latest('gt', j['frame_id'])
        contract = store.latest('input', j['frame_id'])
    else:
        gt_versions = _versions(store, 'gt', j['frame_id'], context)
        input_versions = _versions(store, 'input', j['frame_id'], context)
        gt = gt_versions[0] if gt_versions else None
        contract = input_versions[0] if input_versions else None
    reasons = []
    if not gt:
        reasons.append(_reason('missing_current_gt', 'The current ground truth is missing.'))
    elif gt['status'] != 'approved':
        reasons.append(_reason('current_gt_not_approved', 'The current ground truth is not approved.'))
    if gt and gt.get('content_sha256') != j['ground_truth_sha256']:
        reasons.append(_reason('ground_truth_changed', 'The current ground truth differs from the version graded.'))
    if not contract:
        reasons.append(_reason('missing_current_input', 'The current input is missing.'))
    elif contract.get('input_hash') != j['input_sha256']:
        reasons.append(_reason('input_changed', 'The current input differs from the version graded.'))
    if gt and contract and gt.get('input_hash') != contract.get('input_hash'):
        reasons.append(_reason('current_gt_input_mismatch', 'The current ground truth is not bound to the current input.'))
    if reasons:
        return {'eligible': False, 'status': 'stale', 'reasons': reasons}
    if context is None:
        critical = store.has_critical_issue(j['frame_id'], gt['content_sha256'], contract['input_hash'])
    else:
        if context['critical_issues'] is None:
            context['critical_issues'] = store.critical_issue_bindings()
        critical = (j['frame_id'], gt['content_sha256'], contract['input_hash']) in context['critical_issues']
    if critical:
        return {'eligible': False, 'status': 'ineligible', 'reasons': [
            _reason('critical_reference_issue', 'An unresolved critical reference issue excludes this judgment from current scores.')]}
    return {'eligible': True, 'status': 'judged', 'reasons': []}


def _bound_versions(store, j, context=None):
    inputs = _versions(store, 'input', j['frame_id'], context)
    contract = next((x for x in inputs if x.get('input_hash') == j['input_sha256']), None)
    references = [x for x in _versions(store, 'gt', j['frame_id'], context)
                  if x.get('content_sha256') == j['ground_truth_sha256'] and x.get('input_hash') == j['input_sha256']]
    gt = next((x for x in references if x.get('revision') == j.get('gt_revision')), None)
    if gt is None:
        gt = next((x for x in references if x.get('status') == 'approved'), None)
    return contract, gt


def _score(j, gt, rubric):
    if not gt:
        return None, None, 'The bound ground-truth version is unavailable; the saved score is preserved only in the raw judgment.'
    if not rubric:
        return None, None, 'The exact bound rubric is unavailable; no default scoring policy is substituted.'
    try:
        scores = score_judgment({k: j[k] for k in _SCORE_FIELDS if k in j},
                                dict(gt['content'], status='approved'), rubric=(rubric or {}).get('content'))
        index = benchmark_index(scores, rubric['content']) if rubric else None
        return scores, index, None
    except (KeyError, ValueError, TypeError) as exc:
        return None, None, 'Unable to reproduce the bound score: ' + str(exc)


def _scored_eligibility(store, j, scores, error, context=None):
    eligibility = _eligibility(store, j, context)
    if error or (scores and not scores.get('evaluable')):
        return {'eligible': False, 'status': 'ineligible', 'reasons': eligibility['reasons'] + [
            _reason('score_unavailable', error or scores.get('reason') or 'This reference is unscorable.')]}
    return eligibility


def _presentation_view(text, fields):
    """Display a single complete JSON code fence without altering strict checks.

    No extraction from prose, concatenated blocks or malformed JSON is allowed.
    analyze_output retains duplicate-key defects instead of choosing a value.
    """
    match = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```\s*', text)
    if not match or '```' in match.group(1):
        return None
    view = analyze_output(match.group(1), fields)
    return view if view['json_valid'] else None


def _prediction(store, run_id, frame_id, contract=None, expected_hash=None):
    row = store.db.execute('SELECT data FROM predictions WHERE run_id=? AND frame_id=?', (run_id, frame_id)).fetchone()
    if not row:
        return None
    prediction = json.loads(row[0])
    if expected_hash and (prediction.get('prediction_sha256') != expected_hash or
                         hashlib.sha256(prediction.get('output', '').encode()).hexdigest() != expected_hash):
        return None
    from .output_format import expected_output_fields
    from .adapted_inputs import archived_output_fields
    own_hash = prediction.get('input_sha256') or prediction.get('input_hash')
    if own_hash:
        source = next((x for x in store.versions('input', frame_id) if x.get('input_hash') == own_hash), None)
    else:
        source = None
    if source:
        fields = expected_output_fields(source)
    elif contract and prediction.get('track') == 'historical':
        fields = archived_output_fields(contract)
    else:
        fields = expected_output_fields(contract)
    text = prediction.get('output', prediction.get('output_text', '')) or ''
    view = analyze_output(text, fields)
    presentation = _presentation_view(text, fields) if not view['json_valid'] else None
    result = dict(prediction, text=text, output_view=view)
    if presentation is not None:
        result['presentation_view'] = presentation
    return result


def _image_entries(store, contract):
    if not contract:
        return []
    if contract.get('input_images'):
        return [dict(entry) for entry in contract['input_images']]
    sha = contract.get('image_sha256')
    if not isinstance(sha, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', sha):
        return []
    # Canonical images are stored by content hash. The current local_images row
    # may have been replaced, so recover only the exact old content-addressed file.
    current = store.get_records('local_images', contract['frame_id'])
    path = next((x['path'] for x in current if x.get('sha256') == sha), None)
    if path is None:
        path = next(('objects/' + sha + suffix for suffix in ('.jpg', '.jpeg', '.png', '.webp')
                     if (store.root / ('objects/' + sha + suffix)).is_file()), None)
    if path is None:
        path = 'objects/' + sha + '.jpg'
    mime = 'image/png' if path.lower().endswith('.png') else 'image/webp' if path.lower().endswith('.webp') else 'image/jpeg'
    return [{'attachment_id': 'canonical', 'path': path, 'sha256': sha, 'media_type': mime}]


def _read_image(store, entry):
    path = (store.root / entry['path']).resolve()
    if not path.is_relative_to(store.root) or not path.is_file():
        raise KeyError('The exact image used for this judgment is unavailable.')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry.get('sha256'):
        raise ValueError('The bound image checksum does not match; no replacement image is shown.')
    media_type = entry.get('media_type', 'image/jpeg')
    if media_type not in ('image/jpeg', 'image/png', 'image/webp', 'image/gif'):
        raise ValueError('Unsupported bound image media type')
    return raw, media_type


def _images(store, contract, prefix):
    result = []
    for entry in _image_entries(store, contract):
        image = dict(entry, url=prefix + '/' + quote(entry['attachment_id'], safe=''), available=True)
        try:
            _read_image(store, entry)
        except (KeyError, ValueError, OSError) as exc:
            image.update(available=False, error=str(exc))
        result.append(image)
    return result


def _warnings(run, prediction, contract):
    warnings = ['Scores are provisional; judge calibration has not been established.']
    if run.get('track') == 'historical':
        warnings.append('Historical output: a matching frame does not verify the original request/response pairing.')
    own_hash = (prediction or {}).get('input_sha256') or (prediction or {}).get('input_hash')
    if own_hash and contract and own_hash != contract['input_hash']:
        warnings.append('The saved candidate request hash differs from the input used by the judge.')
    if not own_hash:
        warnings.append('The candidate output has no verified frozen input hash; original prompt correspondence is unverified.')
    return warnings


def build_review(store, run_id='', group_id=''):
    judgments, rubrics, runs = _judgment_headers(store), _rubrics(store), store.list_runs()
    context = {'versions': {}, 'critical_issues': None}
    groups = {}
    for j in judgments:
        group = _group(j, rubrics.get(j['rubric_sha256']))
        bucket = groups.setdefault(group['group_id'], dict(group, run_ids=[], count=0))
        bucket['count'] += 1
        if j['run_id'] not in bucket['run_ids']:
            bucket['run_ids'].append(j['run_id'])
    judgment_counts = Counter(j['run_id'] for j in judgments)
    runs = [dict(r, judgment_count=judgment_counts[r['id']]) for r in runs]
    if run_id and not any(r['id'] == run_id for r in runs):
        raise KeyError('Unknown prediction run')
    selected_run = next((r for r in runs if r['id'] == run_id), None) if run_id else next(
        (r for r in runs if r['judgment_count']), runs[0] if runs else None)
    run_id = selected_run['id'] if selected_run else None
    candidates = [g for g in groups.values() if run_id in g['run_ids']]
    if group_id and not any(g['group_id'] == group_id for g in candidates):
        raise KeyError('Unknown judgment configuration for this run')
    selected_group = next((g for g in candidates if g['group_id'] == group_id), None) if group_id else (candidates[0] if candidates else None)
    selected_rows = [j for j in _judgments(store, run_id)
                     if selected_group and _group(j)['group_id'] == selected_group['group_id']] if run_id else []
    rows_by_frame = {}
    for j in selected_rows:
        rows_by_frame.setdefault(j['frame_id'], []).append(j)
    # The list needs only status; full answer parsing belongs in the detail view.
    predictions = {row['frame_id']: {'status': row['status']} for row in store.db.execute(
        "SELECT frame_id,json_extract(data,'$.status') AS status FROM predictions WHERE run_id=?",
        (run_id,))} if run_id else {}
    rows, counts = [], {'judged': 0, 'unjudged': 0, 'stale': 0, 'ineligible': 0}
    for fid in sorted(set(predictions) | set(rows_by_frame)):
        pred, options = predictions.get(fid), rows_by_frame.get(fid, [])
        # Show the current eligible revision first, otherwise the latest retained
        # assessment; all immutable assessment IDs remain accessible in details.
        options.sort(key=lambda j: (_eligibility(store, j, context)['eligible'], j.get('created_at', ''), j['id']), reverse=True)
        j = options[0] if options else None
        scores, index = None, None
        if j:
            contract, gt = _bound_versions(store, j, context)
            scores, index, error = _score(j, gt, rubrics.get(j['rubric_sha256']))
            eligibility = _scored_eligibility(store, j, scores, error, context)
        elif pred and pred.get('status') == 'succeeded':
            eligibility = {'eligible': False, 'status': 'unjudged', 'reasons': [
                _reason('not_judged', 'No judgment exists for this frame and selected configuration.')]}
        else:
            eligibility = {'eligible': False, 'status': 'ineligible', 'reasons': [
                _reason('prediction_not_succeeded', 'A successful model answer is required before judging.')]}
        frame = store.frame(fid)
        rows.append({'frame_id': fid, 'recording_id': frame.get('recording_id'), 'tags': frame.get('tags', []),
                     'prediction_status': (pred or {}).get('status', 'missing'), 'status': eligibility['status'],
                     'judgment_id': j['id'] if j else None, 'judgment_ids': [x['id'] for x in options],
                     'eligible': eligibility['eligible'], 'eligibility': eligibility, 'exclusion_reasons': eligibility['reasons'],
                     'scores': scores, 'benchmark_index': index['value'] if index else None})
        counts[eligibility['status']] += 1
    counts.update(total=len(rows), judgments=len(selected_rows), predictions=len(predictions))
    return {'schema_version': SCHEMA, 'runs': runs, 'groups': list(groups.values()), 'frames': rows,
            'selected_run_id': run_id, 'selected_group_id': selected_group['group_id'] if selected_group else None,
            'counts': counts, 'notice': 'Read-only score inspection. Each judgment is shown with its original input, reference and rubric. Stale or blocked judgments are excluded from current model scores.'}


def review_detail(store, judgment_id):
    row = store.db.execute('SELECT data FROM judgments WHERE id=?', (judgment_id,)).fetchone()
    if not row:
        raise KeyError('Unknown judgment')
    j = json.loads(row[0])
    run = next((r for r in store.list_runs() if r['id'] == j['run_id']), None)
    if not run:
        raise KeyError('The judgment run is unavailable')
    rubric = _rubrics(store).get(j['rubric_sha256'])
    contract, gt = _bound_versions(store, j)
    prediction = _prediction(store, j['run_id'], j['frame_id'], contract, j['prediction_sha256'])
    scores, index, error = _score(j, gt, rubric)
    eligibility = _scored_eligibility(store, j, scores, error)
    warnings = _warnings(run, prediction, contract)
    if error:
        warnings.append(error)
    if not contract:
        warnings.append('The exact input version used by this judge is unavailable; the current input is not substituted.')
    if not prediction:
        warnings.append('The exact model answer matching this judgment is unavailable; a different answer is not substituted.')
    facts = {f['id']: f for f in (gt or {}).get('content', {}).get('facts', [])}
    graded = (scores or {}).get('graded_details', {})
    claims = [dict(c, facts=[facts[fid] for fid in c.get('fact_ids', []) if fid in facts])
              for c in graded.get('claims', j.get('claims', []))]
    coverage = graded.get('fact_coverage', j.get('fact_coverage'))
    if coverage is None:
        coverage = [{'fact_id': f['id'], 'coverage': 1 if f['id'] in j.get('supported_fact_ids', []) else 0,
                     'claim_ids': [], 'evidence': 'Legacy binary coverage; no graded coverage rationale was recorded.'}
                    for f in facts.values() if f.get('required')]
    criteria_meta = {c['id']: c for c in (rubric or {}).get('content', {}).get('analytic_criteria', [])}
    criteria = [dict(criteria_meta.get(key, {}), id=key, **value) for key, value in j.get('criterion_assessments', {}).items()]
    images = _images(store, contract, '/api/scoring-review/' + quote(judgment_id, safe='') + '/image')
    if any(not img['available'] for img in images):
        warnings.append('Some original evidence images are unavailable or fail checksum verification.')
    return {'schema_version': SCHEMA, 'frame': store.frame(j['frame_id']), 'run': run,
            'group': _group(j, rubric), 'judgment': j, 'prediction': prediction, 'ground_truth': gt,
            'input': contract, 'rubric': rubric, 'images': images, 'scores': scores,
            'benchmark_index': index['value'] if index else None, 'index_calculation': index, 'index_details': index,
            'eligibility': eligibility, 'claims': claims,
            'fact_coverage': [dict(c, fact=facts.get(c['fact_id'])) for c in coverage],
            'criteria': criteria, 'warnings': warnings,
            'bindings': {key: j.get(key) for key in ('input_sha256', 'ground_truth_sha256', 'prediction_sha256', 'rubric_sha256', 'gt_revision')}}


def prediction_detail(store, run_id, frame_id):
    run = next((r for r in store.list_runs() if r['id'] == run_id), None)
    if not run:
        raise KeyError('Unknown prediction run')
    prediction = _prediction(store, run_id, frame_id)
    if prediction is None:
        raise KeyError('No prediction exists for this frame and run')
    own_hash = prediction.get('input_sha256') or prediction.get('input_hash')
    contract = next((x for x in store.versions('input', frame_id) if own_hash and x.get('input_hash') == own_hash), None)
    prefix = '/api/scoring-review/prediction-image/' + quote(run_id, safe='') + '/' + quote(frame_id, safe='')
    return {'schema_version': SCHEMA, 'frame': store.frame(frame_id), 'run': run, 'group': None,
            'judgment': None, 'prediction': prediction, 'ground_truth': None, 'input': contract, 'rubric': None,
            'images': _images(store, contract, prefix), 'scores': None, 'benchmark_index': None, 'index_calculation': None, 'index_details': None,
            'claims': [], 'fact_coverage': [], 'criteria': [], 'bindings': {'input_sha256': own_hash},
            'eligibility': {'eligible': False, 'status': 'unjudged' if prediction['status'] == 'succeeded' else 'ineligible',
                            'reasons': [_reason('no_selected_judgment', 'No judgment is selected; this is the saved model answer only.')]},
            'warnings': ['No scoring judgment is selected. No current ground truth is presented as graded evidence.'] + _warnings(run, prediction, contract)}


def review_image(store, judgment_id, attachment_id):
    row = store.db.execute('SELECT data FROM judgments WHERE id=?', (judgment_id,)).fetchone()
    if not row:
        raise KeyError('Unknown judgment')
    contract, _ = _bound_versions(store, json.loads(row[0]))
    entry = next((x for x in _image_entries(store, contract) if x['attachment_id'] == attachment_id), None)
    if not entry:
        raise KeyError('No matching image in this judgment input')
    return _read_image(store, entry)


def prediction_image(store, run_id, frame_id, attachment_id):
    prediction = _prediction(store, run_id, frame_id)
    if not prediction:
        raise KeyError('Unknown prediction')
    own_hash = prediction.get('input_sha256') or prediction.get('input_hash')
    contract = next((x for x in store.versions('input', frame_id) if own_hash and x.get('input_hash') == own_hash), None)
    entry = next((x for x in _image_entries(store, contract) if x['attachment_id'] == attachment_id), None)
    if not entry:
        raise KeyError('No verified input image for this prediction')
    return _read_image(store, entry)
