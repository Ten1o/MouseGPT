"""Offline end-to-end scientific workflow tests using isolated fixture records."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mousebench.api_providers import execute_job
from mousebench.protocol import rubric_hash
from mousebench.store import Conflict, FIELDS, Store
from mousebench.workflow import Workflow


FRAMES = ('240420_C10M1_00003972', '240420_C10M1_00003973', '240421_C11M1_00003974')


def default_rubric():
    """Keep these legacy workflow fixtures bound to their original arithmetic.

    Graded v2 ingestion and calibration have independent end-to-end tests.
    """
    return json.loads((Path(__file__).resolve().parents[1] / 'protocol' / 'rubric-legacy-v2.json').read_text())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.import_catalog({'frames': [{'id': fid, 'recording_id': fid.rsplit('_', 1)[0]} for fid in FRAMES],
                                   'assets': [], 'ratings': [], 'descriptions': []})
        for fid in FRAMES:
            self.store.register_image(fid, b'\xff\xd8\xffoffline-fixture-' + fid.encode(), 'test fixture only')
        self.workflow = Workflow(self.store)
        self.content = {'reference_description': 'Fixture mouse facing left.',
                        'sections': {field: 'Fixture description.' for field in FIELDS},
                        'facts': [{'id': 'f1', 'text': 'Mouse faces left.', 'field': 'Head',
                                   'evidence': 'image', 'required': True}], 'uncertainties': []}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def inputs(self, frames=None):
        return self.workflow.freeze_inputs({'frame_ids': list(frames or FRAMES), 'prompt': 'Describe the six MouseGPT fields.',
                                            'system_prompt': 'Use only visible evidence.', 'image_detail': 'high', 'reviewer': 'Fixture researcher'})

    def frozen_rubric(self):
        saved = self.workflow.save_rubric({'content': default_rubric(), 'author': 'Fixture researcher'})
        return self.workflow.freeze_rubric({'id': saved['id'], 'reviewer': 'Fixture researcher'})

    def batch(self, task, frames=None, **options):
        payload = {'task': task, 'executor': 'codex', 'model': 'fixture-assistant', 'frame_ids': list(frames or [FRAMES[0]])}
        payload.update(options)
        return self.workflow.create_batch(payload)

    def receipt(self, job_id, result, **extra):
        value = {'job_id': job_id, 'worker': 'Fixture Codex', 'model_version': 'fixture-v1', 'result': result}
        value.update(extra)
        return value

    def draft(self, fid=FRAMES[0], issues=None):
        batch = self.batch('ground_truth', [fid])
        job_id = batch['jobs'][0]['id']
        result = {'content': copy.deepcopy(self.content), 'issues': issues or []}
        self.workflow.complete(self.receipt(job_id, result))
        return self.store.latest('gt', fid)

    def approve(self, fid=FRAMES[0]):
        gt = self.store.latest('gt', fid)
        content = gt['content'] if gt else copy.deepcopy(self.content)
        reviewed = self.store.save_gt(fid, {'status': 'reviewed', 'content': content,
                            'reviewer': 'Fixture human', 'expected_revision': gt['revision'] if gt else 0})
        return self.store.save_gt(fid, {'status': 'approved', 'content': content, 'reviewer': 'Fixture human',
                            'expected_revision': reviewed['revision'], 'human_confirmed': True})

    def candidate(self, frames=None, executor='codex'):
        batch = self.batch('inference', frames, executor=executor,
                           provider_config={'api_key_env': 'MOUSEBENCH_FIXTURE_KEY'} if executor != 'codex' else None)
        claimed = self.workflow.claim(batch['batch']['id'], 'Fixture Codex', 395)
        for job in claimed['jobs']:
            result = {'output_text': '\n'.join(field + ': Fixture answer.' for field in FIELDS)}
            if executor != 'codex':
                result['request_log'] = {'kind': 'native_execution', 'input_sha256': job['packet']['input_contract_sha256'],
                                         'job_payload_sha256': job['packet']['payload_sha256'], 'observed_model': 'fixture-v1'}
            self.workflow.complete(self.receipt(job['id'], result, lease_token=job['lease_token']))
        return batch['batch']['run_id']

    def judgment(self):
        return {'claims': [{'text': 'Mouse faces left.', 'label': 'supported', 'evidence': 'Visible head orientation.', 'fact_ids': ['f1']}],
                'supported_fact_ids': ['f1'], 'structure_score': 1, 'instruction_score': .5, 'issues': [],
                'criterion_assessments': {criterion['id']: {'verdict': 'met', 'evidence': 'Synthetic fixture evidence only.'}
                                          for criterion in default_rubric().get('analytic_criteria', [])}}

    def test_bulk_input_freeze_skip_and_atomic_invalid_frame(self):
        result = self.inputs()
        self.assertEqual(result['saved'], 3)
        initial = [self.store.latest('input', fid) for fid in FRAMES]
        self.assertEqual(self.inputs()['skipped'], 3)
        self.assertEqual([self.store.latest('input', fid) for fid in FRAMES], initial)
        with self.assertRaises((KeyError, ValueError)):
            self.workflow.freeze_inputs({'frame_ids': [FRAMES[0], 'not-a-frame'], 'prompt': 'Changed', 'reviewer': 'Fixture human', 'overwrite': True})
        self.assertEqual(self.store.latest('input', FRAMES[0]), initial[0])
        with self.assertRaises(ValueError):
            self.workflow.freeze_inputs({'prompt': 'Changed', 'reviewer': 'Fixture human',
                                         'velocity': {'value': 1, 'units': 'cm/s', 'source': 'fixture'}})

    def test_rubric_versions_are_saved_and_frozen_immutably(self):
        content = default_rubric()
        draft = self.workflow.save_rubric({'content': content, 'author': 'Fixture author'})
        frozen = self.workflow.freeze_rubric({'id': draft['id'], 'reviewer': 'Fixture human'})
        self.assertEqual(frozen['status'], 'frozen')
        self.assertEqual(frozen['sha256'], rubric_hash(content))
        self.assertEqual(frozen['calibration_status'], 'not_established')
        self.assertEqual(self.workflow.save_rubric({'content': content, 'author': 'Another'})['id'], frozen['id'])
        content['rules'].append('An explanatory change creates a new version.')
        newer = self.workflow.save_rubric({'content': content, 'author': 'Fixture author', 'parent_id': frozen['id']})
        self.assertNotEqual(newer['id'], frozen['id'])
        self.assertEqual(newer['status'], 'draft')
        self.assertEqual(self.workflow.rubric(frozen['id']), frozen)

    def test_codex_completion_is_a_draft_with_ai_provenance_and_no_approval(self):
        self.inputs()
        with patch('mousebench.api_providers.execute_job') as network:
            batch = self.batch('ground_truth', FRAMES)
            packets = self.workflow.export_jobs(batch['batch']['id'])['jobs']
            self.assertEqual(len(packets), 3)
            result = {'content': self.content, 'issues': []}
            self.workflow.complete(self.receipt(packets[0]['job_id'], result))
        network.assert_not_called()
        gt = self.store.latest('gt', FRAMES[0])
        self.assertEqual(gt['status'], 'draft')
        self.assertTrue(gt['provenance']['assistant_generated'])
        self.assertEqual(gt['provenance']['actor_type'], 'assistant')
        self.assertEqual(gt['provenance']['job_id'], packets[0]['job_id'])
        self.assertFalse(gt['provenance']['human_confirmed'])
        self.assertEqual(self.workflow.batch(batch['batch']['id'])['batch']['counts']['succeeded'], 1)
        self.assertEqual(len(self.workflow.export_jobs(batch['batch']['id'])['jobs']), 2)

    def test_independent_check_issues_require_human_resolution_before_approval(self):
        self.inputs()
        self.draft()
        batch = self.batch('gt_check')
        packet = self.workflow.job(batch['jobs'][0]['id'])['packet']
        data = json.loads(packet['payload_template']['messages'][1]['content'])
        self.assertIn('existing_reference', data)
        self.assertNotIn('candidate_descriptions', data)
        self.assertNotIn('archived_expert_evidence', data)
        issue = {'severity': 'critical', 'category': 'contradiction', 'message': 'Fixture ambiguity needs a human decision.', 'fact_ids': ['f1']}
        self.workflow.complete(self.receipt(packet['job_id'], {'issues': [issue]}))
        gt = self.store.latest('gt', FRAMES[0])
        self.assertEqual(gt['status'], 'draft')
        with self.assertRaises(ValueError):
            self.store.save_gt(FRAMES[0], {'status': 'reviewed', 'actor_type': 'assistant', 'reviewer': 'AI',
                                         'content': gt['content'], 'expected_revision': gt['revision']})
        reviewed = self.store.save_gt(FRAMES[0], {'status': 'reviewed', 'content': gt['content'], 'reviewer': 'Human', 'expected_revision': gt['revision']})
        approval = {'status': 'approved', 'content': reviewed['content'], 'reviewer': 'Human', 'expected_revision': reviewed['revision']}
        with self.assertRaises(ValueError):
            self.store.save_gt(FRAMES[0], approval)
        with self.assertRaises(ValueError):
            self.store.save_gt(FRAMES[0], dict(approval, human_confirmed=True))
        current_issue = self.workflow.issues(current_only=True)[0]
        self.workflow.resolve_issue({'id': current_issue['id'], 'status': 'resolved', 'reviewer': 'Human',
                                     'resolution': 'Inspected the fixture image and resolved the interpretation.'})
        approved = self.store.save_gt(FRAMES[0], dict(approval, human_confirmed=True))
        self.assertEqual(approved['status'], 'approved')
        self.assertTrue(approved['provenance']['assistant_generated'])
        self.assertTrue(approved['provenance']['human_confirmed'])

    def test_mocked_api_inference_produces_controlled_native_record(self):
        self.inputs()
        batch = self.batch('inference', executor='openai', model='fixture-model',
                           provider_config={'api_key_env': 'MOUSEBENCH_FIXTURE_KEY', 'settings': {'max_output_tokens': 500}})
        job = self.workflow.claim(batch['batch']['id'], 'Fixture Codex')['jobs'][0]
        envelope = job['packet']['expected_result_schema']
        self.assertEqual(envelope['type'], 'object')
        self.assertEqual(envelope['required'], ['output_text'])
        self.assertEqual(envelope['properties']['output_text']['type'], 'string')
        calls = []
        def transport(url, headers, payload, timeout):
            calls.append(payload)
            return {'id': 'resp_fixture', 'model': 'fixture-snapshot', 'status': 'completed',
                    'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Overall: Fixture mouse.'}]}]}
        with patch.dict(os.environ, {'MOUSEBENCH_FIXTURE_KEY': 'fixture-key-not-real'}):
            result = execute_job(job['packet'], batch['batch']['provider_config'], self.temp.name, transport)
        self.workflow.complete(self.receipt(job['id'], result, lease_token=job['lease_token'], model_version='fixture-snapshot'))
        predictions = self.store.predictions(batch['batch']['run_id'])
        self.assertEqual(len(calls), 1)
        contract = self.store.latest('input', FRAMES[0])
        self.assertEqual(calls[0]['input'][0]['content'][0]['text'], contract['system_prompt'])
        self.assertEqual(calls[0]['input'][1]['content'][0]['text'], contract['prompt'])
        self.assertNotIn('Local submission envelope', json.dumps(calls[0]))
        self.assertNotIn('text', calls[0])
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]['output_text'], 'Overall: Fixture mouse.')
        self.assertEqual(predictions[0]['request_log']['kind'], 'native_execution')
        self.assertEqual(predictions[0]['request_log']['observed_model'], 'fixture-snapshot')
        self.assertNotIn('base64', json.dumps(predictions[0]['request_log']))
        self.assertNotIn('fixture-key-not-real', json.dumps(predictions[0]))
        self.assertEqual(self.store.list_runs()[0]['track'], 'controlled')

    def test_api_inference_requires_claim_and_native_log(self):
        self.inputs()
        batch = self.batch('inference', executor='openai', provider_config={'api_key_env': 'MOUSEBENCH_FIXTURE_KEY'})
        result = {'output_text': 'Fixture answer'}
        with self.assertRaises(Conflict):
            self.workflow.complete(self.receipt(batch['jobs'][0]['id'], result))
        job = self.workflow.claim(batch['batch']['id'], 'Fixture Codex')['jobs'][0]
        with self.assertRaises(ValueError):
            self.workflow.complete(self.receipt(job['id'], result, lease_token=job['lease_token']))
        self.assertEqual(self.store.predictions(batch['batch']['run_id']), [])
        self.assertEqual(self.workflow.job(job['id'])['status'], 'leased')

    def test_judgment_import_and_composite_keep_coverage_explicit(self):
        self.inputs()
        rubric = self.frozen_rubric()
        self.approve()
        run_id = self.candidate()
        batch = self.batch('judge', run_id=run_id, rubric_id=rubric['id'])
        self.workflow.complete(self.receipt(batch['jobs'][0]['id'], self.judgment()))
        cards = self.workflow.scorecards()
        self.assertEqual(len(cards['models']), 1)
        card = cards['models'][0]
        self.assertAlmostEqual(card['benchmark_index'], 95)
        self.assertEqual(card['index_count'], 1)
        self.assertAlmostEqual(card['benchmark_coverage'], 1 / 3)
        self.assertAlmostEqual(card['completion_adjusted_index'], 95 / 3)
        self.assertEqual(card['calibration_reports'], [])
        group = cards['groups'][0]
        self.assertEqual(group['input_track'], 'image_only')
        self.assertEqual(group['rubric_sha256'], rubric['sha256'])

    def test_completion_is_idempotent_and_rejects_conflicting_replay(self):
        self.inputs()
        batch = self.batch('ground_truth')
        receipt = self.receipt(batch['jobs'][0]['id'], {'content': self.content, 'issues': []})
        self.workflow.complete(receipt)
        duplicate = self.workflow.complete(receipt)
        self.assertTrue(duplicate['duplicate'])
        self.assertEqual(len(self.store.versions('gt', FRAMES[0])), 1)
        changed = copy.deepcopy(receipt)
        changed['result']['content']['reference_description'] = 'Changed answer'
        with self.assertRaises(Conflict):
            self.workflow.complete(changed)
        self.assertEqual(len(self.store.versions('gt', FRAMES[0])), 1)

    def test_stale_input_or_reference_rejects_assisted_results_without_data_write(self):
        self.inputs()
        batch = self.batch('ground_truth')
        self.store.save_input(FRAMES[0], {'prompt': 'Changed task', 'reviewer': 'Human', 'expected_revision': 1})
        with self.assertRaises(Conflict):
            self.workflow.complete(self.receipt(batch['jobs'][0]['id'], {'content': self.content, 'issues': []}))
        self.assertIsNone(self.store.latest('gt', FRAMES[0]))
        self.assertEqual(self.workflow.job(batch['jobs'][0]['id'])['status'], 'pending')
        batch = self.batch('ground_truth')
        self.draft()
        with self.assertRaises(Conflict):
            self.workflow.complete(self.receipt(batch['jobs'][0]['id'], {'content': self.content, 'issues': []}))
        self.assertEqual(len(self.store.versions('gt', FRAMES[0])), 1)

    def test_judge_result_rejects_reference_changed_after_job_creation(self):
        self.inputs()
        self.frozen_rubric()
        approved = self.approve()
        run_id = self.candidate()
        batch = self.batch('judge', run_id=run_id)
        changed = copy.deepcopy(self.content)
        changed['facts'][0]['text'] = 'The mouse faces right instead.'
        self.store.save_gt(FRAMES[0], {'status': 'draft', 'reviewer': 'Human', 'expected_revision': approved['revision'], 'content': changed})
        self.approve()
        with self.assertRaises((Conflict, ValueError)):
            self.workflow.complete(self.receipt(batch['jobs'][0]['id'], self.judgment()))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM judgments').fetchone()[0], 0)

    def test_multiresult_import_rolls_back_all_earlier_successes_on_error(self):
        self.inputs()
        batch = self.batch('ground_truth', FRAMES[:2])
        receipts = [self.receipt(batch['jobs'][0]['id'], {'content': self.content, 'issues': []}),
                    self.receipt(batch['jobs'][1]['id'], {'content': self.content, 'issues': [{'severity': 'impossible', 'message': 'Bad issue'}]})]
        with self.assertRaises(ValueError):
            self.workflow.submit_results({'batch_id': batch['batch']['id'], 'records': receipts})
        self.assertEqual(self.workflow.batch(batch['batch']['id'])['batch']['counts']['pending'], 2)
        self.assertIsNone(self.store.latest('gt', FRAMES[0]))
        self.assertIsNone(self.store.latest('gt', FRAMES[1]))
        self.assertEqual(self.workflow.issues(), [])

    def test_lease_is_exclusive_and_interruption_requires_explicit_retry_ack(self):
        self.inputs()
        batch = self.batch('ground_truth')
        claimed = self.workflow.claim(batch['batch']['id'], 'Worker A')['jobs'][0]
        self.assertEqual(self.workflow.claim(batch['batch']['id'], 'Worker B')['jobs'], [])
        with self.assertRaises(Conflict):
            self.workflow.complete(self.receipt(claimed['id'], {'content': self.content, 'issues': []}, lease_token=claimed['lease_token']))
        retry = {'batch_id': batch['batch']['id'], 'job_ids': [claimed['id']], 'reviewer': 'Human'}
        with self.assertRaises(ValueError):
            self.workflow.retry_jobs(retry)
        self.assertEqual(self.workflow.retry_jobs(dict(retry, acknowledge_uncertain=True))['pending'], 1)
        second = self.workflow.claim(batch['batch']['id'], 'Worker A')['jobs'][0]
        self.assertNotEqual(second['lease_token'], claimed['lease_token'])
        self.assertEqual(second['attempts'], 2)
        self.workflow.fail({'job_id': second['id'], 'lease_token': second['lease_token'], 'uncertain': True, 'code': 'transport_timeout'})
        self.assertEqual(self.workflow.job(second['id'])['status'], 'uncertain')
        self.assertEqual(self.workflow.retry_jobs({'batch_id': batch['batch']['id'], 'reviewer': 'Human'})['pending'], 0)
        with self.assertRaises(ValueError):
            self.workflow.retry_jobs(retry)

    def test_release_freezes_reference_and_rejects_changed_bindings(self):
        self.inputs()
        rubric = self.frozen_rubric()
        self.approve()
        with self.assertRaises(ValueError):
            self.workflow.create_release({'name': 'Fixture release', 'reviewer': 'Human', 'rubric_id': rubric['id']})
        release = self.workflow.create_release({'name': 'Fixture subset', 'reviewer': 'Human', 'rubric_id': rubric['id'],
                                               'frame_ids': [FRAMES[0]]})
        self.assertTrue(release['complete'])
        self.assertEqual(release['scientific_validation'], 'provisional')
        self.assertEqual(release['rubric']['sha256'], rubric['sha256'])
        before = copy.deepcopy(release)
        self.batch('inference', release_id=release['id'])
        self.store.save_input(FRAMES[0], {'prompt': 'Changed', 'reviewer': 'Human', 'expected_revision': 1})
        with self.assertRaises(ValueError):
            self.batch('inference', release_id=release['id'])
        self.assertEqual(self.workflow.record('release', release['id']), before)

    def test_calibration_binds_existing_outputs_rubric_input_and_recording(self):
        self.inputs()
        rubric = self.frozen_rubric()
        run_id = self.candidate()
        prediction = self.store.predictions(run_id)[0]
        contract = self.store.latest('input', FRAMES[0])
        record = {'id': 'calibration_fixture', 'frame_id': FRAMES[0], 'recording_id': 'untrusted supplied recording',
                  'verified_by': 'Fixture human', 'human_verified': True, 'split': 'development', 'kind': 'correctness',
                  'human_label': 'correct', 'judge_label': 'correct', 'input_sha256': contract['input_hash'],
                  'prediction_sha256': prediction['prediction_sha256'], 'rubric_sha256': rubric['sha256'],
                  'judge': {'provider': 'codex', 'model': 'fixture-assistant', 'version': 'fixture-v1'}}
        report = self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': [record]})
        self.assertEqual(report['records'][0]['recording_id'], '240420_C10M1')
        self.assertEqual(report['rubric_sha256'], rubric['sha256'])
        for key in ('prediction_sha256', 'rubric_sha256', 'input_sha256'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': [dict(record, **{key: 'f' * 64})]})
        with self.assertRaises(ValueError):
            self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': [dict(record, human_verified=False)]})

    def calibration_fixture(self):
        self.inputs()
        rubric = self.frozen_rubric()
        self.approve()
        run_id = self.candidate()
        batch = self.batch('judge', run_id=run_id, rubric_id=rubric['id'])
        self.workflow.complete(self.receipt(batch['jobs'][0]['id'], self.judgment()))
        return rubric, run_id, self.workflow.calibration_template(run_id)

    def label_calibration(self, records):
        labeled = copy.deepcopy(records)
        for record in labeled:
            record.update(split='heldout', human_verified=True, verified_by='Independent fixture human',
                          human_label='contradicted' if record['kind'] == 'claim' else .25)
        return labeled

    def test_managed_calibration_template_is_blind_with_complete_sensory_context(self):
        rubric, run_id, template = self.calibration_fixture()
        self.assertTrue(template['blinded'])
        self.assertEqual(len(template['records']), 3)
        self.assertEqual(self.workflow.calibration_template(run_id), template)
        self.assertNotIn('fixture-assistant', json.dumps(template))
        self.assertNotIn('Fixture researcher', json.dumps(template))
        prediction = self.store.predictions(run_id)[0]
        contract = self.store.latest('input', FRAMES[0])
        image = self.store.image(FRAMES[0])
        for record in template['records']:
            self.assertNotIn('judge', record)
            self.assertNotIn('judge_label', record)
            self.assertIsNone(record['split'])
            self.assertIsNone(record['human_label'])
            self.assertFalse(record['human_verified'])
            self.assertEqual(record['verified_by'], '')
            self.assertTrue(record['id'].startswith('managed_calibration_'))
            self.assertEqual(record['context']['candidate_text'], prediction['output_text'])
            self.assertEqual(record['context']['canonical_input']['prompt'], contract['prompt'])
            self.assertEqual(record['context']['canonical_input']['system_prompt'], contract['system_prompt'])
            self.assertEqual(record['context']['image'], {'path': image['path'], 'sha256': image['sha256']})
            self.assertNotIn('source_path', record['context']['image'])
            self.assertNotIn('approved_reference', record['context'])
            self.assertEqual(record['rubric_sha256'], rubric['sha256'])

    def test_blind_human_labels_join_authoritative_judge_verdicts_after_submission(self):
        rubric, run_id, template = self.calibration_fixture()
        records = self.label_calibration(template['records'])
        before = copy.deepcopy(records)
        imported = self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': records})
        stored_judgment = json.loads(self.store.db.execute('SELECT data FROM judgments WHERE run_id=?', (run_id,)).fetchone()[0])
        self.assertEqual(records, before)
        self.assertEqual(imported['report']['n'], 3)
        self.assertEqual(len(imported['report']['disagreement_ids']), 3)
        for record in imported['records']:
            expected = 'supported' if record['kind'] == 'claim' else stored_judgment[record['dimension']]
            self.assertEqual(record['judge_label'], expected)
            self.assertEqual(record['judge'], stored_judgment['judge'])
            self.assertEqual(record['human_label'], 'contradicted' if record['kind'] == 'claim' else .25)
        self.assertEqual(self.workflow.calibration_template(run_id), template)

    def test_managed_calibration_rejects_verdict_binding_and_context_tampering_atomically(self):
        rubric, _, template = self.calibration_fixture()
        records = self.label_calibration(template['records'])
        edits = {
            'judge_label': 'unverifiable', 'judge': {'provider': 'other', 'model': 'other', 'version': 'other'},
            'claim_id': 'c99', 'claim_text': 'A replacement claim.', 'dimension': 'instruction_score',
            'id': 'renamed-managed-row', 'kind': 'correctness', 'prediction_sha256': 'f' * 64,
            'input_sha256': 'f' * 64, 'rubric_sha256': 'f' * 64, 'ground_truth_sha256': 'f' * 64,
            'managed_binding': dict(records[0]['managed_binding'], judgment_sha256='f' * 64),
            'context': dict(records[0]['context'], candidate_text='An easier replacement answer.'),
        }
        for key, value in edits.items():
            altered = copy.deepcopy(records[0])
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': [records[1], altered]})
            self.assertEqual(self.workflow.records('calibration'), [])
        no_binding = copy.deepcopy(records[0])
        del no_binding['managed_binding']
        with self.assertRaises(ValueError):
            self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': [no_binding]})
        self.assertEqual(self.workflow.records('calibration'), [])

    def test_managed_calibration_requires_explicit_split_and_independent_human_labels(self):
        rubric, _, template = self.calibration_fixture()
        with self.assertRaises(ValueError):
            self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': template['records']})
        ready = self.label_calibration(template['records'])[0]
        for key, value in (('split', None), ('human_verified', False), ('verified_by', ''), ('human_label', None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': [dict(ready, **{key: value})]})
        self.assertEqual(self.workflow.records('calibration'), [])

    def test_managed_calibration_rejects_a_reference_changed_after_export(self):
        rubric, run_id, template = self.calibration_fixture()
        gt = self.store.latest('gt', FRAMES[0])
        content = copy.deepcopy(gt['content'])
        content['reference_description'] = 'A corrected reference.'
        self.store.save_gt(FRAMES[0], {'status': 'draft', 'reviewer': 'Human', 'content': content, 'expected_revision': gt['revision']})
        self.approve()
        self.assertEqual(self.workflow.calibration_template(run_id)['records'], [])
        with self.assertRaises(ValueError):
            self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': self.label_calibration(template['records'])})
        self.assertEqual(self.workflow.records('calibration'), [])

    def test_external_fully_specified_pairwise_calibration_remains_supported(self):
        self.inputs()
        rubric = self.frozen_rubric()
        first_run = self.candidate()
        second = self.batch('inference')
        self.workflow.complete(self.receipt(second['jobs'][0]['id'], {'output_text': 'A different fixture answer.'}))
        first = self.store.predictions(first_run)[0]
        other = self.store.predictions(second['batch']['run_id'])[0]
        record = {'id': 'historical_human_pair', 'frame_id': FRAMES[0], 'kind': 'pairwise', 'split': 'development',
                  'input_sha256': self.store.latest('input', FRAMES[0])['input_hash'],
                  'prediction_sha256': first['prediction_sha256'], 'other_prediction_sha256': other['prediction_sha256'],
                  'rubric_sha256': rubric['sha256'], 'judge': {'provider': 'external', 'model': 'historic', 'version': '1'},
                  'judge_label': 'b', 'human_label': 'a', 'human_verified': True, 'verified_by': 'Fixture human'}
        imported = self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': [record]})
        self.assertEqual(imported['report']['n'], 1)
        self.assertEqual(imported['records'][0]['judge_label'], 'b')
        self.assertEqual(imported['records'][0]['human_label'], 'a')
        self.assertNotIn('managed_binding', imported['records'][0])

    def test_release_bound_judge_cannot_substitute_new_reference_or_rubric(self):
        self.inputs()
        rubric = self.frozen_rubric()
        approved = self.approve()
        run_id = self.candidate()
        release = self.workflow.create_release({'name': 'Fixed fixture', 'reviewer': 'Human', 'frame_ids': [FRAMES[0]], 'rubric_id': rubric['id']})
        self.batch('judge', run_id=run_id, rubric_id=rubric['id'], release_id=release['id'])
        changed_rubric = copy.deepcopy(rubric['content'])
        changed_rubric['rules'].append('A different immutable rubric.')
        newer = self.workflow.save_rubric({'content': changed_rubric, 'author': 'Human', 'parent_id': rubric['id']})
        self.workflow.freeze_rubric({'id': newer['id'], 'reviewer': 'Human'})
        with self.assertRaises(ValueError):
            self.batch('judge', run_id=run_id, rubric_id=newer['id'], release_id=release['id'])
        changed_gt = copy.deepcopy(self.content)
        changed_gt['facts'][0]['text'] = 'A different reference claim.'
        self.store.save_gt(FRAMES[0], {'status': 'draft', 'reviewer': 'Human', 'content': changed_gt, 'expected_revision': approved['revision']})
        self.approve()
        with self.assertRaises(ValueError):
            self.batch('judge', run_id=run_id, rubric_id=rubric['id'], release_id=release['id'])

    def test_stale_issues_and_judgments_do_not_silently_apply_to_new_gt(self):
        self.inputs()
        self.frozen_rubric()
        self.draft(issues=[{'severity': 'warning', 'message': 'Inspect the original fixture interpretation.'}])
        approved = self.approve()
        run_id = self.candidate()
        batch = self.batch('judge', run_id=run_id)
        self.workflow.complete(self.receipt(batch['jobs'][0]['id'], self.judgment()))
        self.assertEqual(len(self.workflow.issues(current_only=True)), 1)
        changed_gt = copy.deepcopy(self.content)
        changed_gt['reference_description'] = 'A newly edited fixture reference.'
        self.store.save_gt(FRAMES[0], {'status': 'draft', 'reviewer': 'Human', 'content': changed_gt, 'expected_revision': approved['revision']})
        self.assertEqual(self.workflow.issues(current_only=True), [])
        self.assertEqual(len(self.workflow.issues()), 1)
        cards = self.workflow.scorecards()
        self.assertEqual(cards['stale_judgments_excluded'], 1)
        self.assertEqual(cards['models'], [])

    def test_calibration_rejects_old_output_rebound_to_changed_input(self):
        self.inputs()
        rubric = self.frozen_rubric()
        run_id = self.candidate()
        prediction = self.store.predictions(run_id)[0]
        contract = self.store.save_input(FRAMES[0], {'prompt': 'A different task.', 'reviewer': 'Human', 'expected_revision': 1})
        record = {'id': 'stale_fixture', 'frame_id': FRAMES[0], 'verified_by': 'Human', 'human_verified': True,
                  'split': 'development', 'kind': 'correctness', 'human_label': 'correct', 'judge_label': 'correct',
                  'input_sha256': contract['input_hash'], 'prediction_sha256': prediction['prediction_sha256'],
                  'rubric_sha256': rubric['sha256'], 'judge': {'provider': 'codex', 'model': 'fixture', 'version': 'v1'}}
        with self.assertRaises(ValueError):
            self.workflow.import_calibration({'rubric_id': rubric['id'], 'records': [record]})
        self.assertEqual(self.workflow.records('calibration'), [])

    def test_custom_index_weights_are_calculated_and_reported(self):
        self.inputs()
        content = default_rubric()
        content['scoring']['index'].update(base=.6, structure_weight=.25, instruction_weight=.15)
        rubric = self.workflow.save_rubric({'content': content, 'author': 'Human'})
        self.workflow.freeze_rubric({'id': rubric['id'], 'reviewer': 'Human'})
        self.approve()
        run_id = self.candidate()
        batch = self.batch('judge', run_id=run_id)
        self.workflow.complete(self.receipt(batch['jobs'][0]['id'], self.judgment()))
        card = self.workflow.scorecards()['models'][0]
        self.assertAlmostEqual(card['benchmark_index'], 92.5)
        self.assertEqual(card['index_weights'], {'base': .6, 'structure_weight': .25, 'instruction_weight': .15})
        self.assertEqual(card['index_formula'], content['scoring']['index']['formula'])

    def test_api_common_receipt_validation_applies_to_gt_and_checks(self):
        self.inputs()
        batch = self.batch('ground_truth', executor='openai', model='fixture-model',
                           provider_config={'api_key_env': 'MOUSEBENCH_FIXTURE_KEY'})
        job = self.workflow.claim(batch['batch']['id'], 'Fixture Codex')['jobs'][0]
        def transport(*args):
            return {'id': 'response-fixture', 'model': 'fixture-snapshot', 'status': 'completed',
                    'output_text': json.dumps({'content': self.content, 'issues': []})}
        with patch.dict(os.environ, {'MOUSEBENCH_FIXTURE_KEY': 'fixture-not-real'}):
            valid = execute_job(job['packet'], batch['batch']['provider_config'], self.temp.name, transport)
        for key, wrong in (('provider', 'azure'), ('model', 'other-model'), ('job_payload_sha256', 'e' * 64),
                           ('input_sha256', 'f' * 64), ('settings', {}), ('api_mode', 'chat_completions'),
                           ('endpoint', 'https://wrong.example.com/v1/responses')):
            result = copy.deepcopy(valid)
            result['request_log'][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.workflow.complete(self.receipt(job['id'], result, lease_token=job['lease_token'], model_version='fixture-snapshot'))
        with self.assertRaises(ValueError):
            self.workflow.complete(self.receipt(job['id'], valid, lease_token=job['lease_token'], model_version='incorrect-version'))
        self.assertIsNone(self.store.latest('gt', FRAMES[0]))
        self.workflow.complete(self.receipt(job['id'], valid, lease_token=job['lease_token'], model_version='fixture-snapshot'))
        self.assertEqual(self.store.latest('gt', FRAMES[0])['status'], 'draft')

    def test_batch_rejects_actual_model_version_drift(self):
        self.inputs()
        batch = self.batch('ground_truth', FRAMES[:2])
        result = {'content': self.content, 'issues': []}
        self.workflow.complete(self.receipt(batch['jobs'][0]['id'], result, model_version='snapshot-A'))
        with self.assertRaises(Conflict):
            self.workflow.complete(self.receipt(batch['jobs'][1]['id'], result, model_version='snapshot-B'))
        self.assertIsNone(self.store.latest('gt', FRAMES[1]))
        self.assertEqual(self.workflow.job(batch['jobs'][1]['id'])['status'], 'pending')
        self.assertEqual(self.workflow.record('batch', batch['batch']['id'])['observed_model_version'], 'snapshot-A')

    def test_new_judge_claims_require_explicit_fact_mapping(self):
        self.inputs()
        self.frozen_rubric()
        self.approve()
        run_id = self.candidate()
        batch = self.batch('judge', run_id=run_id)
        missing = self.judgment()
        del missing['claims'][0]['fact_ids']
        with self.assertRaises(ValueError):
            self.workflow.complete(self.receipt(batch['jobs'][0]['id'], missing))
        inconsistent = self.judgment()
        inconsistent['claims'][0]['fact_ids'] = []
        with self.assertRaises(ValueError):
            self.workflow.complete(self.receipt(batch['jobs'][0]['id'], inconsistent))
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM judgments').fetchone()[0], 0)

    def test_same_warning_promoted_to_critical_cannot_escape_approval_gate(self):
        self.inputs()
        batch = self.batch('ground_truth')
        content = copy.deepcopy(self.content)
        content['uncertainties'] = ['Direction is ambiguous.']
        issue = {'severity': 'critical', 'category': 'contradiction', 'message': 'Direction is ambiguous.',
                 'fact_ids': ['f1'], 'evidence': 'Fixture ambiguity.'}
        self.workflow.complete(self.receipt(batch['jobs'][0]['id'], {'content': content, 'issues': [issue]}))
        issues = self.workflow.issues(current_only=True)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]['severity'], 'critical')
        self.assertEqual(issues[0]['history'][-1]['event'], 'severity_promoted')
        with self.assertRaises(ValueError):
            self.approve()

    def test_judge_packet_includes_saved_reference_paragraph(self):
        self.inputs()
        self.frozen_rubric()
        self.approve()
        run_id = self.candidate()
        batch = self.batch('judge', run_id=run_id)
        packet = self.workflow.job(batch['jobs'][0]['id'])['packet']
        data = json.loads(packet['payload_template']['messages'][1]['content'])
        self.assertEqual(data['approved_reference']['reference_description'], self.content['reference_description'])
        self.assertNotIn('reviewer', data['approved_reference'])

    def test_changed_judge_generation_configuration_produces_separate_score_groups(self):
        self.inputs()
        self.frozen_rubric()
        self.approve()
        run_id = self.candidate()
        judge_versions = []
        for temperature in (0, 1):
            batch = self.batch('judge', run_id=run_id, executor='openai', model='judge-model',
                               provider_config={'api_key_env': 'MOUSEBENCH_FIXTURE_KEY', 'settings': {'temperature': temperature}})
            job = self.workflow.claim(batch['batch']['id'], 'Fixture Codex')['jobs'][0]
            with patch.dict(os.environ, {'MOUSEBENCH_FIXTURE_KEY': 'fixture-not-real'}):
                result = execute_job(job['packet'], batch['batch']['provider_config'], self.temp.name,
                                     lambda *args: {'model': 'same-snapshot', 'output_text': json.dumps(self.judgment())})
            self.workflow.complete(self.receipt(job['id'], result, lease_token=job['lease_token'], model_version='same-snapshot'))
            judge_versions.append(batch['batch']['execution_config_sha256'])
        self.assertNotEqual(judge_versions[0], judge_versions[1])
        cards = self.workflow.scorecards()
        self.assertEqual(len(cards['groups']), 2)
        self.assertEqual(len(cards['models']), 2)
        self.assertEqual(len({group['judge']['version'] for group in cards['groups']}), 2)

    def test_scorecards_report_workflow_failures_and_uncertain_attempts(self):
        self.inputs()
        self.frozen_rubric()
        self.approve()
        batch = self.batch('inference', FRAMES)
        jobs = self.workflow.claim(batch['batch']['id'], 'Fixture Codex', 3)['jobs']
        self.workflow.complete(self.receipt(jobs[0]['id'], {'output_text': 'Fixture answer.'}, lease_token=jobs[0]['lease_token']))
        self.workflow.fail({'job_id': jobs[1]['id'], 'lease_token': jobs[1]['lease_token'], 'code': 'request_rejected'})
        self.workflow.fail({'job_id': jobs[2]['id'], 'lease_token': jobs[2]['lease_token'], 'code': 'transport_timeout', 'uncertain': True})
        judge = self.batch('judge', run_id=batch['batch']['run_id'])
        self.workflow.complete(self.receipt(judge['jobs'][0]['id'], self.judgment()))
        card = self.workflow.scorecards()['models'][0]
        self.assertEqual(card['failed_count'], 1)
        self.assertEqual(card['uncertain_count'], 1)
        self.assertEqual(card['planned_count'], 3)
        self.assertEqual(card['attempted_count'], 3)
        self.assertEqual(card['execution_attempts'], 3)
        self.assertEqual(card['pending_count'], 0)
        self.assertEqual(card['leased_count'], 0)
        self.assertEqual(card['index_count'], 1)

    def test_critical_issue_found_after_approval_blocks_judging_until_resolved(self):
        self.inputs()
        self.frozen_rubric()
        self.approve()
        run_id = self.candidate()
        first = self.batch('judge', run_id=run_id)
        self.workflow.complete(self.receipt(first['jobs'][0]['id'], self.judgment()))
        pending = self.batch('judge', run_id=run_id)
        self.assertEqual(len(self.workflow.scorecards()['models']), 1)
        check = self.batch('gt_check')
        issue = {'severity': 'critical', 'category': 'contradiction', 'message': 'A late independent check found a potentially unsupported required fact.', 'fact_ids': ['f1']}
        self.workflow.complete(self.receipt(check['jobs'][0]['id'], {'issues': [issue]}))
        self.assertEqual(self.store.latest('gt', FRAMES[0])['status'], 'approved')
        self.assertEqual(self.workflow.scorecards()['models'], [])
        with self.assertRaises(ValueError):
            self.batch('judge', run_id=run_id)
        with self.assertRaises(ValueError):
            self.workflow.complete(self.receipt(pending['jobs'][0]['id'], self.judgment()))
        current = self.workflow.issues(current_only=True)[0]
        self.workflow.resolve_issue({'id': current['id'], 'status': 'resolved', 'reviewer': 'Human',
                                     'resolution': 'Re-examined the input and verified the existing reference fact.'})
        self.assertEqual(len(self.workflow.scorecards()['models']), 1)
        self.workflow.complete(self.receipt(pending['jobs'][0]['id'], self.judgment()))
        self.assertEqual(len(self.workflow.scorecards()['models']), 1)

    def test_critical_candidate_issues_preserve_reference_and_other_model_scores(self):
        self.inputs()
        rubric = self.frozen_rubric()
        approved = self.approve()
        run_ids = []
        for explicit_target in (False, True):
            run_id = self.candidate()
            run_ids.append(run_id)
            batch = self.batch('judge', run_id=run_id)
            result = self.judgment()
            result['claims'] = [{'text': 'Mouse faces right.', 'label': 'contradicted',
                                 'evidence': 'The fixture reference and image show the opposite.', 'fact_ids': []}]
            result['supported_fact_ids'] = []
            issue = {'severity': 'critical', 'category': 'contradiction',
                     'message': 'Candidate contains a critical factual error.'}
            if explicit_target:
                issue['target'] = 'candidate'
            result['issues'] = [issue]
            self.workflow.complete(self.receipt(batch['jobs'][0]['id'], result))
            self.assertFalse(self.store.has_critical_issue(FRAMES[0], approved['content_sha256'], approved['input_hash']))
        issues = self.workflow.issues(current_only=True)
        self.assertEqual(len(issues), 2)
        self.assertTrue(all(i['target'] == 'candidate' and i['status'] == 'open' for i in issues))
        other_run = self.candidate()
        other = self.batch('judge', run_id=other_run)
        self.workflow.complete(self.receipt(other['jobs'][0]['id'], self.judgment()))
        cards = {card['run_id']: card for card in self.workflow.scorecards()['models']}
        self.assertEqual(len(cards), 3)
        for run_id in run_ids:
            self.assertEqual(cards[run_id]['benchmark_index'], 0)
            self.assertEqual(cards[run_id]['index_count'], 1)
        self.assertGreater(cards[other_run]['benchmark_index'], 0)
        self.assertEqual(self.store.latest('gt', FRAMES[0]), approved)
        release = self.workflow.create_release({'frame_ids': [FRAMES[0]], 'rubric_id': rubric['id'],
                                               'reviewer': 'Fixture human', 'name': 'Candidate issues do not alter GT'})
        self.assertTrue(release['complete'])

    def test_judge_reference_issue_blocks_shared_reference_until_human_resolution(self):
        self.inputs()
        rubric = self.frozen_rubric()
        self.approve()
        run_id = self.candidate()
        other_run = self.candidate()
        other = self.batch('judge', run_id=other_run)
        self.workflow.complete(self.receipt(other['jobs'][0]['id'], self.judgment()))
        batch = self.batch('judge', run_id=run_id)
        result = self.judgment()
        result['issues'] = [{'target': 'reference', 'severity': 'critical', 'category': 'reference_contradiction',
                             'message': 'The approved reference may contradict the visible image.', 'fact_ids': ['f1']}]
        self.workflow.complete(self.receipt(batch['jobs'][0]['id'], result))
        issue = self.workflow.issues(current_only=True)[0]
        self.assertEqual(issue['target'], 'reference')
        self.assertEqual(self.workflow.scorecards()['models'], [])
        with self.assertRaises(ValueError):
            self.batch('judge', run_id=other_run)
        with self.assertRaises(ValueError):
            self.workflow.create_release({'frame_ids': [FRAMES[0]], 'rubric_id': rubric['id'],
                                          'reviewer': 'Fixture human', 'name': 'Blocked reference'})
        self.workflow.resolve_issue({'id': issue['id'], 'status': 'resolved', 'reviewer': 'Fixture human',
                                     'resolution': 'Rechecked the visible image; original reference is correct.'})
        self.assertEqual(len(self.workflow.scorecards()['models']), 2)
        self.assertEqual(len(self.batch('judge', run_id=other_run)['jobs']), 1)

    def test_ground_truth_tasks_cannot_downgrade_reference_issues_to_candidate(self):
        self.inputs()
        for task in ('ground_truth', 'gt_check'):
            with self.subTest(task=task):
                batch = self.batch(task)
                result = {'issues': [{'target': 'candidate', 'severity': 'critical', 'category': 'contradiction',
                                      'message': 'Reference problem from ' + task, 'fact_ids': ['f1']}]}
                if task == 'ground_truth':
                    result['content'] = copy.deepcopy(self.content)
                self.workflow.complete(self.receipt(batch['jobs'][0]['id'], result))
        issues = self.workflow.issues(current_only=True)
        self.assertEqual(len(issues), 2)
        self.assertTrue(all(i['target'] == 'reference' for i in issues))
        with self.assertRaises(ValueError):
            self.approve()

    def test_scorecard_issue_index_matches_individual_checks_and_refreshes(self):
        self.inputs(); self.frozen_rubric()
        for fid in FRAMES:
            self.approve(fid)
        run = self.candidate(FRAMES)
        batch = self.batch('judge', FRAMES, run_id=run)
        for job in batch['jobs']:
            self.workflow.complete(self.receipt(job['id'], self.judgment()))

        def issue(iid, fid, **changes):
            gt = self.store.latest('gt', fid)
            value = dict(id=iid, frame_id=fid, status='open', severity='critical',
                         content_sha256=gt['content_sha256'], input_sha256=gt['input_hash'])
            value.update(changes)
            with self.store.transaction():
                self.workflow.put('issue', value)

        issue('stale-content', FRAMES[0], content_sha256='a'*64)
        issue('candidate-only', FRAMES[0], target='candidate')
        issue('warning-only', FRAMES[0], severity='warning')
        issue('resolved', FRAMES[1], status='resolved')
        issue('stale-input', FRAMES[1], input_sha256='b'*64)
        issue('current', FRAMES[2])  # Missing target retains reference default.

        store = self.store
        class IndividualChecks:
            def __contains__(self, binding):
                return store.has_critical_issue(*binding)

        def assert_parity(blocked):
            self.store.db.execute('PRAGMA query_only=ON')
            try:
                with patch.object(self.store, 'critical_issue_bindings', return_value=IndividualChecks()):
                    expected = self.workflow.scorecards()
                actual = self.workflow.scorecards()
                self.assertEqual(actual, expected)
                self.assertEqual(actual['critical_issue_judgments_excluded'], blocked)
            finally:
                self.store.db.execute('PRAGMA query_only=OFF')

        assert_parity(1)
        issue('resolved', FRAMES[1], status='open')
        assert_parity(2)
        issue('current', FRAMES[2], status='resolved')
        assert_parity(1)


if __name__ == '__main__':
    unittest.main()
