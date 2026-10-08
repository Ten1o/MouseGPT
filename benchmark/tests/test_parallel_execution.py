"""Offline tests for concurrent requests and serial, resumable persistence."""
import os
import threading
import unittest
from unittest.mock import patch

from mousebench.parallel_execution import run_batches
import tests.test_workflow as fixtures


class ParallelExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.fixture.inputs()
        self.wf = self.fixture.workflow
        self.env = patch.dict(os.environ, {'MOUSEBENCH_FIXTURE_KEY': 'offline-fixture-key'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.fixture.tearDown()

    def batch(self, model='fixture-model', frames=None):
        return self.fixture.batch('inference', frames or fixtures.FRAMES[:2], executor='openai', model=model,
            provider_config={'api_key_env': 'MOUSEBENCH_FIXTURE_KEY'})['batch']['id']

    @staticmethod
    def response():
        return {'id': 'offline-response', 'model': 'fixture-snapshot', 'status': 'completed',
                'output_text': '```json\n{"Overall":"Original raw fixture output"}\n```'}

    def test_two_batches_overlap_and_resume_without_repeating(self):
        batches = [self.batch('fixture-a'), self.batch('fixture-b')]
        barrier = threading.Barrier(4)
        thread_ids = set()
        callback_ids = []
        def transport(*args):
            thread_ids.add(threading.get_ident())
            barrier.wait(timeout=5)
            return self.response()
        result = run_batches(self.wf, batches, workers=4, transport=transport,
            on_progress=lambda event: callback_ids.append(threading.get_ident()))
        self.assertEqual(result['attempted'], 4)
        self.assertEqual(len(thread_ids), 4)
        self.assertEqual(set(callback_ids), {threading.get_ident()})
        self.assertEqual(len(self.wf.records('receipt')), 4)
        self.assertTrue(all(b['counts']['succeeded'] == 2 for b in result['batches']))
        for b in result['batches']:
            self.assertEqual([p['output_text'] for p in self.fixture.store.predictions(b['run_id'])],
                             [self.response()['output_text']] * 2)
        with patch('mousebench.parallel_execution.execute_job') as execute:
            resumed = run_batches(self.wf, batches, workers=4)
        execute.assert_not_called()
        self.assertEqual(resumed['attempted'], 0)

    def test_failure_is_uncertain_and_does_not_replay(self):
        batch = self.batch(frames=[fixtures.FRAMES[0]])
        def timeout(*args):
            raise TimeoutError('offline timeout')
        result = run_batches(self.wf, [batch], workers=1, transport=timeout)
        self.assertEqual(result['batches'][0]['counts']['uncertain'], 1)
        self.assertEqual(result['results'][0]['error']['code'], 'transport_timeout')
        with patch('mousebench.parallel_execution.execute_job') as execute:
            self.assertEqual(run_batches(self.wf, [batch])['attempted'], 0)
        execute.assert_not_called()

    def test_receipt_survives_ingestion_failure_and_progress_callback_failure(self):
        batch = self.batch(frames=[fixtures.FRAMES[0]])
        with patch.object(self.wf, 'complete', side_effect=ValueError('offline import failure')):
            result = run_batches(self.wf, [batch], workers=1, transport=lambda *args: self.response(),
                on_progress=lambda event: 1 / 0)
        self.assertEqual(result['batches'][0]['counts']['failed'], 1)
        self.assertEqual(len(self.wf.records('receipt')), 1)
        self.assertTrue(result['observer_errors'])
        self.assertEqual(self.wf.ingest_receipts(batch)['imported'], 1)
        self.assertEqual(self.wf.batch(batch, False)['batch']['counts']['succeeded'], 1)

    def test_dynamic_workers_preserve_job_identity_for_identical_requests(self):
        batches = [self.batch('same-model'), self.batch('same-model')]
        control = {'workers': 1}
        lock = threading.Lock()
        calls = []
        barrier = threading.Barrier(3)
        events = []
        def factory(job):
            def transport(*args):
                with lock:
                    calls.append(job['id'])
                    first = len(calls) == 1
                if not first:
                    barrier.wait(timeout=5)
                return dict(self.response(), output_text=job['id'])
            return transport
        def progress(event):
            events.append(event)
            if event['phase'] == 'finished' and event['finished'] == 1:
                control['workers'] = 3
        result = run_batches(self.wf, batches, workers=1, worker_ceiling=3,
                             concurrency_control=lambda: control['workers'],
                             transport_factory=factory, on_progress=progress)
        self.assertEqual(result['attempted'], 4)
        self.assertEqual(result['workers'], 3)
        self.assertEqual(len(set(calls)), 4)
        self.assertTrue(any(e['phase'] == 'concurrency_changed' and e['workers'] == 3 for e in events))
        for item in result['results']:
            job = self.wf.job(item['job_id'])
            self.assertEqual(job['status'], 'succeeded')
            self.assertEqual(job['raw_result']['output_text'], job['id'])

    def test_unprepared_job_requeued_midrun_is_never_claimed(self):
        batch = self.batch()
        jobs = self.wf.batch(batch)['jobs']
        excluded, prepared = jobs[0]['id'], jobs[1]['id']
        job = self.wf.job(excluded)
        with self.fixture.store.transaction():
            job.update(status='failed', error={'code': 'offline_previous_failure'})
            self.wf.save_job(job)
        def progress(event):
            if event['phase'] == 'starting':
                self.wf.retry_jobs({'batch_id': batch, 'job_ids': [excluded], 'reviewer': 'offline concurrent retry'})
        result = run_batches(self.wf, [batch], workers=1, allowed_job_ids=[prepared],
                             transport=lambda *args: self.response(), on_progress=progress)
        self.assertEqual(result['attempted'], 1)
        self.assertEqual(result['results'][0]['job_id'], prepared)
        self.assertEqual(self.wf.job(prepared)['status'], 'succeeded')
        self.assertEqual(self.wf.job(excluded)['status'], 'pending')
        self.assertEqual(self.wf.job(excluded)['attempts'], 0)

    def test_import_failure_continues_prepared_jobs_only_when_opted_in(self):
        for opted_in in (False, True):
            with self.subTest(continue_after_import_error=opted_in):
                batch = self.batch('fixture-continue-' + str(opted_in))
                first = self.wf.batch(batch)['jobs'][0]['id']
                complete = self.wf.complete
                def fail_first(payload):
                    if payload['job_id'] == first:
                        raise ValueError('offline per-item import validation')
                    return complete(payload)
                options = {'continue_after_import_error': True} if opted_in else {}
                with patch.object(self.wf, 'complete', side_effect=fail_first):
                    result = run_batches(self.wf, [batch], workers=1,
                                         transport=lambda *args: self.response(), **options)
                self.assertEqual(result['attempted'], 2 if opted_in else 1)
                counts = result['batches'][0]['counts']
                self.assertEqual(counts['failed'], 1)
                self.assertEqual(counts['succeeded'], 1 if opted_in else 0)
                self.assertEqual(counts['pending'], 0 if opted_in else 1)
                failed = self.wf.job(first)
                self.assertEqual(failed['error']['code'], 'result_requires_review')
                self.assertEqual(failed['attempts'], 1)
                self.assertEqual(failed['raw_result']['output_text'], self.response()['output_text'])
                self.assertEqual(self.wf.record('receipt', first + '_1')['result'], failed['raw_result'])


if __name__ == '__main__':
    unittest.main()
