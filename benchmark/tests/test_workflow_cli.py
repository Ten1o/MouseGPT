"""Worker/CLI resume tests. Every provider transport is an offline fixture."""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import workflow as cli
import tests.test_workflow as fixtures


class WorkflowCLITests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkflowTests()
        self.fixture.setUp()
        self.fixture.inputs()
        self.wf = self.fixture.workflow
        self.env = patch.dict(os.environ, {'MOUSEBENCH_FIXTURE_KEY': 'offline-fixture-key'})
        self.env.start()
        self.calls = []

    def tearDown(self):
        self.env.stop()
        self.fixture.tearDown()

    def api_batch(self, task='inference', frames=None):
        return self.fixture.batch(task, frames or fixtures.FRAMES, executor='openai', model='fixture-model',
                                  provider_config={'api_key_env': 'MOUSEBENCH_FIXTURE_KEY'})['batch']['id']

    def transport(self, url, headers, payload, timeout):
        self.calls.append(payload)
        return {'id': 'fixture-response-' + str(len(self.calls)), 'model': 'fixture-snapshot', 'status': 'completed',
                'output_text': 'Overall: Mouse.\nHead: Left.\nLimb: Visible.\nTorso: Visible.\nOthers: Unknown.\nKeywords: mouse'}

    def test_dry_run_does_not_claim_or_call_or_require_credential(self):
        batch_id = self.api_batch()
        with patch.dict(os.environ, {'MOUSEBENCH_FIXTURE_KEY': ''}), patch('mousebench.api_providers.execute_job') as execute:
            result = cli.run_batch(self.wf, batch_id)
        execute.assert_not_called()
        self.assertEqual(result['mode'], 'dry_run')
        self.assertEqual(result['model_calls'], 0)
        self.assertEqual(result['pending'], 3)
        self.assertFalse(result['provider_status']['credential_configured'])
        self.assertTrue(all(job['attempts'] == 0 for job in self.wf.batch(batch_id)['jobs']))

    def test_execute_resume_never_calls_completed_jobs_twice(self):
        batch_id = self.api_batch()
        first = cli.run_batch(self.wf, batch_id, execute=True, limit=1, interval=0, transport=self.transport)
        self.assertEqual(first['attempted'], 1)
        self.assertEqual(first['batch']['counts']['succeeded'], 1)
        self.assertEqual(first['batch']['counts']['pending'], 2)
        second = cli.run_batch(self.wf, batch_id, execute=True, interval=0, transport=self.transport)
        self.assertEqual(second['attempted'], 2)
        self.assertEqual(second['batch']['counts']['succeeded'], 3)
        third = cli.run_batch(self.wf, batch_id, execute=True, interval=0, transport=self.transport)
        self.assertEqual(third['attempted'], 0)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.wf.records('receipt')), 3)
        self.assertTrue(all(job['attempts'] == 1 for job in self.wf.batch(batch_id)['jobs']))
        self.assertEqual(len(self.fixture.store.predictions(second['batch']['run_id'])), 3)

    def test_saved_receipt_can_be_ingested_after_local_failure_without_new_call(self):
        batch_id = self.api_batch('ground_truth', [fixtures.FRAMES[0]])
        def draft_transport(*args):
            self.calls.append(args[2])
            return {'model': 'fixture-snapshot', 'status': 'completed',
                    'output_text': json.dumps({'content': self.fixture.content, 'issues': []})}
        with patch.object(self.wf, 'complete', side_effect=ValueError('Temporary fixture import failure')):
            result = cli.run_batch(self.wf, batch_id, execute=True, interval=0, transport=draft_transport)
        self.assertEqual(result['attempted'], 1)
        self.assertEqual(result['results'][0]['status'], 'failed')
        self.assertEqual(result['results'][0]['error']['code'], 'result_requires_review')
        self.assertEqual(len(self.wf.records('receipt')), 1)
        self.assertIsNone(self.fixture.store.latest('gt', fixtures.FRAMES[0]))
        with patch('mousebench.api_providers.execute_job') as execute:
            imported = self.wf.ingest_receipts(batch_id)
        execute.assert_not_called()
        self.assertEqual(imported, {'imported': 1, 'errors': [], 'model_calls': 0})
        self.assertEqual(self.fixture.store.latest('gt', fixtures.FRAMES[0])['status'], 'draft')
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.wf.ingest_receipts(batch_id)['imported'], 0)
        self.assertEqual(cli.run_batch(self.wf, batch_id, execute=True, interval=0, transport=draft_transport)['attempted'], 0)
        self.assertEqual(len(self.calls), 1)

    def test_timeout_stops_worker_and_uncertain_attempt_is_not_automatically_retried(self):
        batch_id = self.api_batch()
        def timeout(*args):
            self.calls.append(args[2])
            raise TimeoutError('Fixture timeout, no real request')
        first = cli.run_batch(self.wf, batch_id, execute=True, interval=0, transport=timeout)
        self.assertEqual(first['attempted'], 1)
        self.assertEqual(first['batch']['counts']['uncertain'], 1)
        self.assertEqual(first['batch']['counts']['pending'], 2)
        second = cli.run_batch(self.wf, batch_id, execute=True, interval=0, transport=self.transport)
        self.assertEqual(second['attempted'], 2)
        self.assertEqual(second['batch']['counts']['uncertain'], 1)
        self.assertEqual(second['batch']['counts']['succeeded'], 2)
        self.assertEqual(len(self.calls), 3)
        uncertain = [j for j in self.wf.batch(batch_id)['jobs'] if j['status'] == 'uncertain'][0]
        self.assertEqual(uncertain['attempts'], 1)

    def test_rate_limit_stops_without_sending_remaining_frames(self):
        batch_id = self.api_batch()
        calls = []
        def rate_limit(*args):
            calls.append(True)
            return 429, {'error': 'fixture quota exhausted'}, {}
        result = cli.run_batch(self.wf, batch_id, execute=True, interval=0, transport=rate_limit)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['batch']['counts']['failed'], 1)
        self.assertEqual(result['batch']['counts']['pending'], 2)
        self.assertEqual(result['results'][0]['error']['code'], 'rate_limited')
        self.assertTrue(result['results'][0]['error']['retryable'])

    def test_codex_batches_cannot_be_run_through_api_worker(self):
        batch_id = self.fixture.batch('ground_truth')['batch']['id']
        with self.assertRaises(ValueError):
            cli.run_batch(self.wf, batch_id, execute=True, interval=0, transport=self.transport)
        self.assertEqual(self.calls, [])

    def test_cli_work_export_does_not_claim_if_output_already_exists(self):
        batch_id = self.fixture.batch('ground_truth')['batch']['id']
        path = Path(self.fixture.temp.name) / 'existing.json'
        path.write_text('Existing output must stay intact.')
        with self.assertRaises(ValueError):
            cli.main(['--data', self.fixture.temp.name, 'work', batch_id, '--worker', 'Fixture Codex', '--output', str(path)])
        self.assertEqual(path.read_text(), 'Existing output must stay intact.')
        self.assertEqual(self.wf.batch(batch_id)['batch']['counts']['pending'], 1)
        self.assertEqual(self.wf.batch(batch_id)['jobs'][0]['attempts'], 0)

    def test_cli_run_default_is_dry_and_emit_never_overwrites(self):
        batch_id = self.api_batch()
        with patch('builtins.print') as output, patch('mousebench.api_providers.execute_job') as execute:
            status = cli.main(['--data', self.fixture.temp.name, 'run', batch_id, '--interval', '0'])
        execute.assert_not_called()
        self.assertEqual(status, 0)
        self.assertIn('dry_run', output.call_args[0][0])
        destination = Path(self.fixture.temp.name) / 'output.json'
        with patch('builtins.print'):
            cli.emit({'fixture': True}, destination)
        with self.assertRaises(FileExistsError):
            cli.emit({'fixture': False}, destination)
        self.assertEqual(json.loads(destination.read_text()), {'fixture': True})


if __name__ == '__main__':
    unittest.main()
