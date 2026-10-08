"""Overview counts must not confuse job progress with scientific completion."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from mousebench.store import Store, FIELDS
from mousebench.workflow import Workflow
from mousebench.overview import build_overview

FRAMES = ['240420_C10M1_00003972', '240420_C10M1_00003973', '240421_C11M1_00003974']


class OverviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.wf = Workflow(self.store)
        self.content = {'reference_description': 'Fixture only', 'sections': {f: 'Fixture' for f in FIELDS},
                        'facts': [{'id': 'f1', 'text': 'Fixture fact', 'field': 'Head', 'evidence': 'image', 'required': True}], 'uncertainties': []}

    def tearDown(self):
        self.store.close(); self.temp.cleanup()

    def catalog(self):
        self.store.import_catalog({'frames': [{'id': f, 'recording_id': f.rsplit('_', 1)[0]} for f in FRAMES],
                                   'ratings': [{'id': 'r1', 'frame_id': FRAMES[0]}], 'assets': []})
        for f in FRAMES: self.store.register_image(f, b'fixture-'+f.encode(), '/fixture.jpg')

    def input(self, fid, prompt='Describe fixture'):
        old = self.store.latest('input', fid)
        return self.store.save_input(fid, {'prompt': prompt, 'reviewer': 'Fixture', 'expected_revision': old['revision'] if old else 0})

    def approve(self, fid):
        self.input(fid)
        reviewed = self.store.save_gt(fid, {'status': 'reviewed', 'content': self.content, 'expected_revision': 0, 'reviewer': 'Fixture'})
        return self.store.save_gt(fid, {'status': 'approved', 'content': self.content, 'expected_revision': reviewed['revision'], 'reviewer': 'Fixture', 'human_confirmed': True})

    def test_empty_catalog_has_unknown_fractions_not_perfect_completion(self):
        result = build_overview(self.store, self.wf)
        self.assertEqual(result['dataset']['frames'], 0)
        self.assertIsNone(result['preparation']['input_fraction'])
        self.assertEqual(result['results']['scorecards']['models'], [])
        self.assertFalse(result['protocol']['reliability_certified'])

    def test_reference_states_are_mutually_exclusive_and_stale_approval_excluded(self):
        self.catalog(); self.approve(FRAMES[0]); self.approve(FRAMES[1])
        self.input(FRAMES[1], 'Changed input')
        result = build_overview(self.store, self.wf)
        states = result['preparation']['gt_status_counts']
        self.assertEqual(sum(states.values()), 3)
        self.assertEqual((states['approved'], states['needs_review'], states['not_started']), (1, 1, 1))
        self.assertEqual(result['preparation']['approved_usable'], 1)
        self.assertEqual(result['dataset']['recordings'], 2)

    def test_postapproval_critical_issue_is_visible_as_blocked(self):
        self.catalog(); gt = self.approve(FRAMES[0])
        with self.store.transaction():
            self.wf._issue(FRAMES[0], {'severity': 'critical', 'message': 'Fixture disputed fact'}, 'test', gt, gt['input_hash'])
        result = build_overview(self.store, self.wf)
        self.assertEqual(result['preparation']['approved_blocked'], 1)
        self.assertEqual(result['preparation']['approved_usable'], 0)
        self.assertEqual(result['review']['critical'], 1)
        self.assertEqual(result['review']['queue'][0]['frame_id'], FRAMES[0])

    def test_multiple_passes_are_jobs_not_inflated_frame_completion(self):
        self.catalog(); self.input(FRAMES[0])
        for _ in range(2):
            self.wf.create_batch({'task': 'ground_truth', 'executor': 'codex', 'model': 'fixture', 'frame_ids': [FRAMES[0]]})
        result = build_overview(self.store, self.wf)
        self.assertEqual(result['execution']['total_jobs'], 2)
        self.assertEqual(result['execution']['counts']['pending'], 2)
        self.assertEqual(result['preparation']['approved_usable'], 0)
        self.assertEqual(result['preparation']['inputs_frozen'], 1)

    def test_no_provider_calls_or_database_mutations_during_overview(self):
        self.catalog()
        before = list(self.store.db.iterdump())
        with patch('mousebench.api_providers.execute_job') as model:
            build_overview(self.store, self.wf)
        model.assert_not_called()
        self.assertEqual(list(self.store.db.iterdump()), before)

    def test_historical_outputs_do_not_imply_controlled_input_match(self):
        self.catalog(); self.input(FRAMES[0])
        self.store.import_run({'name': 'Historical fixture', 'model': 'unverified', 'track': 'historical',
                               'records': [{'frame_id': FRAMES[0], 'output': 'Fixture answer', 'status': 'succeeded'}]})
        result = build_overview(self.store, self.wf)
        self.assertEqual(result['preparation']['frames_with_outputs'], 1)
        self.assertEqual(result['results']['run_coverage'][0]['current_input_matched'], 0)
        self.assertEqual(result['results']['run_coverage'][0]['coverage'], 1/3)

    def test_sensory_only_batch_mode_is_bound_to_result_provenance(self):
        self.catalog(); self.input(FRAMES[0])
        batch = self.wf.create_batch({'task': 'ground_truth', 'executor': 'codex', 'model': 'fixture',
                                      'frame_ids': [FRAMES[0]], 'evidence_mode': 'sensory_only'})
        self.assertEqual(batch['batch']['evidence_mode'], 'sensory_only')
        self.wf.complete({'job_id': batch['jobs'][0]['id'], 'worker': 'Fixture AI', 'model_version': 'fixture',
                          'result': {'content': copy.deepcopy(self.content), 'issues': []}})
        gt = self.store.latest('gt', FRAMES[0])
        self.assertEqual(gt['provenance']['evidence_mode'], 'sensory_only')
        self.assertEqual(gt['status'], 'draft')
        with self.assertRaises(ValueError):
            self.wf.create_batch({'task': 'inference', 'executor': 'codex', 'model': 'fixture',
                                  'frame_ids': [FRAMES[0]], 'evidence_mode': 'sensory_only'})


if __name__ == '__main__': unittest.main()
