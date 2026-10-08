"""Explicit parallel API execution with all persistence on the calling thread.

Each request gets one durable lease and one attempt. Successful jobs, failures,
and uncertain deliveries are never replayed automatically. A transport override
must be thread-safe and must preserve the normal provider response contract.
"""
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import uuid

from .api_providers import ProviderExecutionError, _payload, execute_job, provider_status


def run_batches(wf, batch_ids, *, workers=80, per_batch_workers=None, transport=None, on_progress=None,
                worker_ceiling=None, concurrency_control=None, transport_factory=None,
                allowed_job_ids=None, continue_after_import_error=False):
    """Execute pending API jobs now; this function has no implicit dry-run mode.

    ``on_progress(event)`` runs on the caller's thread and receives only status
    metadata, never credentials, prompts, or answers. Observer failures do not
    interrupt receipt persistence. In-flight work finishes after a provider
    rejection; further pending jobs for that batch are left unclaimed.
    ``per_batch_workers`` optionally caps concurrent requests within each batch.
    Job IDs are frozen during preflight; ``allowed_job_ids`` optionally restricts
    them to caller-prepared transport bindings. An opted-in import failure keeps
    its receipt and failed state while other prepared jobs continue.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 512:
        raise ValueError('workers must be an integer between 1 and 512')
    worker_ceiling = workers if worker_ceiling is None else worker_ceiling
    if (isinstance(worker_ceiling, bool) or not isinstance(worker_ceiling, int)
            or not workers <= worker_ceiling <= 512):
        raise ValueError('worker_ceiling must be an integer between workers and 512')
    if concurrency_control is not None and not callable(concurrency_control):
        raise ValueError('concurrency_control must be callable')
    if transport_factory is not None and not callable(transport_factory):
        raise ValueError('transport_factory must be callable')
    if not isinstance(continue_after_import_error, bool):
        raise ValueError('continue_after_import_error must be boolean')
    if allowed_job_ids is not None:
        supplied_ids = list(allowed_job_ids)
        if any(not isinstance(jid, str) or not jid for jid in supplied_ids) or len(supplied_ids) != len(set(supplied_ids)):
            raise ValueError('allowed_job_ids must contain distinct nonempty job IDs')
        allowed_job_ids = set(supplied_ids)
    if per_batch_workers is not None and (
        not isinstance(per_batch_workers, int) or isinstance(per_batch_workers, bool)
        or not 1 <= per_batch_workers <= 512
    ):
        raise ValueError('per_batch_workers must be an integer between 1 and 512, or None')
    batch_ids = list(batch_ids)
    if not batch_ids or len(batch_ids) != len(set(batch_ids)):
        raise ValueError('Provide a nonempty list of distinct batch IDs')

    batches = {}; pending_jobs = {}; prepared_ids = set()
    # Finish local validation of every pending packet before any API request.
    for batch_id in batch_ids:
        batch = wf.record('batch', batch_id)
        if batch['executor'] == 'codex':
            raise ValueError('Codex batches must use work/submit')
        status = provider_status(batch['provider_config'])
        if not status['credential_configured']:
            raise ValueError('Set the configured credential environment variable before execution')
        batches[batch_id] = batch
        jobs = [wf.job(row[0]) for row in wf.db.execute(
            "SELECT id FROM wf_jobs WHERE batch_id=? AND status='pending' ORDER BY frame_id", (batch_id,))]
        if allowed_job_ids is not None:
            jobs = [job for job in jobs if job['id'] in allowed_job_ids]
        pending_jobs[batch_id] = deque(job['id'] for job in jobs)
        prepared_ids.update(pending_jobs[batch_id])
        for job in jobs:
            wf._current(job, batch['task'])
            _payload(job['packet'], batch['provider_config'], wf.store.root)
    if allowed_job_ids is not None and prepared_ids != allowed_job_ids:
        raise ValueError('Every allowed job must be pending in an explicitly selected batch')

    queue = deque(batch_ids)
    active_counts = dict.fromkeys(batch_ids, 0)
    blocked = set(); results = []; observer_errors = []; attempted = 0
    execution_id = 'parallel-' + uuid.uuid4().hex[:12]
    active_limit = workers

    def notify(phase, job=None, result=None):
        if on_progress is None:
            return
        event = {'phase': phase, 'attempted': attempted, 'finished': len(results),
                 'workers': active_limit, 'worker_ceiling': worker_ceiling,
                 'per_batch_workers': per_batch_workers, 'execution_id': execution_id}
        if job:
            event.update(batch_id=job['batch_id'], job_id=job['id'], frame_id=job['frame_id'])
        if result:
            event['status'] = result['status']
        try:
            on_progress(event)
        except Exception:
            observer_errors.append({'phase': phase, 'code': 'progress_callback_failed'})

    def claim_next():
        nonlocal attempted
        # Visit each queued batch at most once. If all are saturated, wait for
        # an in-flight request to finish before trying to refill their slots.
        for _ in range(len(queue)):
            batch_id = queue.popleft()
            if batch_id in blocked or not pending_jobs[batch_id]:
                continue
            if per_batch_workers is not None and active_counts[batch_id] >= per_batch_workers:
                queue.append(batch_id)
                continue
            worker = execution_id + '-' + str(attempted + 1).zfill(4)
            claim = {'jobs': []}
            while pending_jobs[batch_id] and not claim['jobs']:
                exact_id = pending_jobs[batch_id].popleft()
                claim = wf.claim(batch_id, worker, 1, job_id=exact_id)
            if pending_jobs[batch_id]:
                queue.append(batch_id)
            if not claim['jobs']:
                continue
            attempted += 1
            job = claim['jobs'][0]
            active_counts[batch_id] += 1
            notify('claimed', job)
            return job
        return None

    def execute_leased_job(job, config):
        # A per-job closure preserves receipt identity even for byte-identical
        # candidate answers submitted for distinct benchmark runs.
        bound_transport = transport_factory(job) if transport_factory else transport
        return execute_job(job['packet'], config, wf.store.root, transport=bound_transport)

    notify('starting')
    with ThreadPoolExecutor(max_workers=worker_ceiling, thread_name_prefix='mousebench-api') as pool:
        futures = {}
        while queue or futures:
            if concurrency_control is not None:
                try:
                    requested = concurrency_control()
                    if isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= worker_ceiling:
                        raise ValueError('Requested workers exceed configured bounds')
                    if requested != active_limit:
                        active_limit = requested
                        notify('concurrency_changed')
                except Exception:
                    observer_errors.append({'phase': 'concurrency_control', 'code': 'invalid_control_kept_previous_limit'})
            while len(futures) < active_limit:
                job = claim_next()
                if job is None:
                    break
                config = batches[job['batch_id']]['provider_config']
                # No Store/Workflow object is accessed from a worker thread.
                future = pool.submit(execute_leased_job, job, config)
                futures[future] = job
            if not futures:
                break
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                job = futures.pop(future)
                active_counts[job['batch_id']] -= 1
                batch = batches[job['batch_id']]
                failure = None
                try:
                    result = future.result()
                except ProviderExecutionError as exc:
                    failure = {'code': exc.code, 'retryable': exc.retryable, 'uncertain': exc.uncertain}
                    if exc.uncertain or exc.code in (
                        'rate_limited', 'request_rejected', 'missing_credential', 'invalid_credential', 'unsafe_endpoint'
                    ):
                        blocked.add(batch['id'])
                except (ValueError, KeyError, TypeError):
                    failure = {'code': 'local_request_validation', 'retryable': False}
                    blocked.add(batch['id'])
                except Exception:
                    # A worker exception without a definitive receipt cannot
                    # establish that Azure never received the request.
                    failure = {'code': 'worker_failure', 'retryable': False, 'uncertain': True}
                    blocked.add(batch['id'])
                if failure:
                    outcome = wf.fail(dict(failure, job_id=job['id'], lease_token=job['lease_token']))
                else:
                    observed = result['request_log'].get('observed_model') or batch['model'] + ' (snapshot not exposed by provider)'
                    wf.record_receipt(job['id'], result, observed, job['worker'])
                    try:
                        outcome = wf.complete({'job_id': job['id'], 'lease_token': job['lease_token'],
                            'worker': job['worker'], 'model_version': observed, 'result': result})
                    except (ValueError, KeyError, TypeError) as exc:
                        outcome = wf.fail({'job_id': job['id'], 'lease_token': job['lease_token'],
                            'code': 'result_requires_review', 'retryable': False, 'result': result})
                        outcome['review_reason'] = str(exc)
                        if not continue_after_import_error:
                            blocked.add(batch['id'])
                results.append(dict(outcome, batch_id=job['batch_id'], frame_id=job['frame_id']))
                notify('finished', job, outcome)

    notify('completed')
    return {'mode': 'executed', 'execution_id': execution_id, 'workers': active_limit,
            'initial_workers': workers, 'worker_ceiling': worker_ceiling,
            'per_batch_workers': per_batch_workers,
            'continue_after_import_error': continue_after_import_error,
            'attempted': attempted, 'results': results, 'observer_errors': observer_errors,
            'batches': [wf.batch(batch_id, False)['batch'] for batch_id in batch_ids],
            'notice': 'No automatic retries. Reconcile uncertain calls; ingest saved receipts without another request.'}
