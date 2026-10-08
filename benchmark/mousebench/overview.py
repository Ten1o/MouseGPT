"""Read-only operational overview; progress is not scientific certification."""
from collections import Counter
from .store import now


def build_overview(store, workflow):
    frames = store.frame_list()
    total = len(frames)
    counts = dict(store.db.execute('SELECT kind,count(*) FROM records GROUP BY kind'))
    statuses = Counter(f['gt_status'] for f in frames)
    inputs = store.latest_versions('input')
    issues = [i for i in workflow.issues(current_only=True) if i['status'] == 'open']
    severity = Counter(i['severity'] for i in issues)
    critical_frames = {i['frame_id'] for i in issues if i['severity'] == 'critical' and i.get('target','reference')=='reference'}
    blocked = sum(f['gt_status'] == 'approved' and f['id'] in critical_frames for f in frames)
    input_count = sum(v is not None for v in inputs.values())
    batch_rows = [workflow.batch(b['id'], False)['batch'] for b in workflow.records('batch')]
    job_states = ('pending', 'leased', 'succeeded', 'failed', 'uncertain')
    jobs = {key: sum(b['counts'][key] for b in batch_rows) for key in job_states}
    by_task = []
    for task in ('ground_truth', 'gt_check', 'inference', 'judge'):
        task_counts = {key: sum(b['counts'][key] for b in batch_rows if b['task'] == task) for key in job_states}
        by_task.append(dict(task=task, total=sum(task_counts.values()), **task_counts))
    run_coverage = []; frames_with_outputs = set()
    for run in store.list_runs():
        predictions = store.prediction_headers(run['id'])
        succeeded = [p for p in predictions if p['status'] == 'succeeded']
        frames_with_outputs.update(p['frame_id'] for p in succeeded)
        matched = sum(run['track'] == 'controlled' and inputs.get(p['frame_id']) is not None
                      and p.get('input_sha256', p.get('input_hash')) == inputs[p['frame_id']]['input_hash']
                      for p in succeeded)
        run_coverage.append({key: run[key] for key in ('id', 'name', 'model', 'track')} | {
            'succeeded': len(succeeded), 'outputs': len(predictions),
            'failed': sum(p['status'] in ('failed', 'timeout', 'cancelled') for p in predictions),
            'missing': max(0, total - len(predictions)) + sum(p['status'] == 'missing' for p in predictions),
            'coverage': len(succeeded) / total if total else None, 'current_input_matched': matched})
    # Add execution failures which did not produce a prediction record.
    by_run = {b.get('run_id'): b for b in batch_rows if b['task'] == 'inference'}
    for run in run_coverage:
        if run['id'] in by_run:
            batch = by_run[run['id']]
            run['job_counts'] = batch['counts']
            run['failed'] = batch['counts']['failed']
            run['uncertain'] = batch['counts']['uncertain']
    rubrics = workflow.records('rubric')
    calibrations = workflow.records('calibration')
    releases = workflow.records('release')
    scored = workflow.scorecards()
    next_actions = []
    if input_count < total:
        next_actions.append({'page': 'protocol', 'label': 'Import recorded inputs', 'detail': f'{total-input_count} frames still need their original per-frame request from the source log.'})
    if not any(r['status'] == 'frozen' for r in rubrics):
        next_actions.append({'page': 'protocol', 'label': 'Review the proposed rubric', 'detail': 'Review criteria during the GT pilot; freeze before judging or release. Freezing does not validate the judge.'})
    if issues:
        next_actions.append({'page': 'workflow', 'label': 'Review flagged evidence', 'detail': f'{len(issues)} open issues; {severity["critical"]} critical. AI flags are proposals for review.'})
    if statuses['approved'] - blocked < total:
        next_actions.append({'page': 'frames', 'label': 'Build and approve references', 'detail': f'{statuses["approved"]-blocked} of {total} frames have current approval with no open critical issue.'})
    if jobs['failed'] or jobs['uncertain'] or jobs['leased']:
        next_actions.append({'page': 'workflow', 'label': 'Inspect unfinished execution', 'detail': 'Leased jobs may be running or interrupted. Reconcile saved receipts before retrying uncertain calls.'})
    if not scored['models']:
        next_actions.append({'page': 'runs', 'label': 'Evaluate a pilot after GT approval', 'detail': 'Create inference and judge batches. The dashboard does not start model calls.'})
    else:
        next_actions.append({'page': 'runs', 'label': 'Compare matched model results', 'detail': 'Use the same input, reference, rubric and judge configuration; inspect coverage and error diagnostics.'})
    if not calibrations:
        next_actions.append({'page': 'workflow', 'label': 'Calibrate the judge', 'detail': 'Compare exact-output human labels on heldout recordings before making reliability claims.'})
    return {
        'schema_version': 'mousebench.overview.v1', 'generated_at': now(),
        'dataset': {'frames': total, 'recordings': len({f.get('recording_id') or f['id'].rsplit('_', 1)[0] for f in frames}),
                    'local_images': counts.get('local_images', 0), 'assets': counts.get('assets', 0),
                    'expert_rows': counts.get('ratings', 0), 'archived_descriptions': counts.get('descriptions', 0),
                    'raw_records': counts.get('raw_records', 0), 'collection_tags': store.collection_tags()},
        'preparation': {'inputs_frozen': input_count, 'input_fraction': input_count / total if total else None,
                        'gt_status_counts': {k: statuses[k] for k in ('not_started', 'draft', 'reviewed', 'approved', 'needs_review')},
                        'approved_usable': statuses['approved']-blocked, 'approved_blocked': blocked,
                        'frames_with_outputs': len(frames_with_outputs),
                        'frames_with_ratings': sum(f['ratings_count'] > 0 for f in frames)},
        'review': {'open': len(issues), 'critical': severity['critical'], 'warning': severity['warning'], 'info': severity['info'],
                    'affected_frames': len({i['frame_id'] for i in issues}),
                    'queue': [{k: i[k] for k in ('id', 'frame_id', 'severity', 'category', 'message', 'target')}
                              for i in sorted(issues, key=lambda i: (('critical','warning','info').index(i['severity']), i['frame_id']))[:20]]},
        'execution': {'batch_count': len(batch_rows), 'total_jobs': sum(jobs.values()), 'counts': jobs, 'by_task': by_task,
                      'batches': [{k: b[k] for k in ('id', 'name', 'task', 'executor', 'model', 'counts', 'total')} for b in batch_rows]},
        'results': {'scorecards': scored, 'run_coverage': run_coverage},
        'protocol': {'rubric_drafts': sum(r['status']=='draft' for r in rubrics), 'rubric_frozen': sum(r['status']=='frozen' for r in rubrics),
                     'calibration_reports': len(calibrations), 'releases': len(releases), 'reliability_certified': False},
        'next_actions': next_actions,
        'limitations': ['Operational completion is separate from scientific validation.',
                        'Expert source rows include duplicate exports; historical collections are not verified model identities.',
                        'The manually selected frame collections are not established unseen test sets; recordings are not verified independent animals.',
                        'Displayed image counts are registered local assets, not a fresh integrity scan.',
                        'Scores are provisional and conditional on evaluated coverage. No cross-configuration leaderboard is implied.']}
