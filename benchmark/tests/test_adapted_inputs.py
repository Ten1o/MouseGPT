"""Prompt standardization creates new inputs without rewriting source evidence."""
import copy
import hashlib
import json
import unittest

from mousebench.adapted_inputs import archived_output_fields
from mousebench.api_providers import _payload, validate_config
from mousebench.output_format import expected_output_fields
from mousebench.providers import build_job_preview
from mousebench.recorded_inputs import ADAPTED_PROTOCOL, input_semantics, verify_input_images
from mousebench.store import FIELDS, digest, dumps
from mousebench.workflow import Workflow
import test_reconstructed_inputs as fixtures
from test_reconstructed_inputs import TEMPLATE, TARGET

STANDARD = '240419_C11M4_00003972'
STANDARD_SYSTEM = '  Fixture v1.0: return Overall, Head, Limb, Torso, Others, Keywords. Tail belongs in Limb.\n'


class AdaptedInputTests(unittest.TestCase):
    def setUp(self):
        # Reuse the archive/reconstruction fixture, without inheriting its tests.
        self.fixture = fixtures.ReconstructedInputTests()
        self.fixture.setUp()
        self.store = self.fixture.store
        self.source = self.fixture.save()
        self.store.import_catalog({'frames': [{'id': STANDARD, 'tags': ['v1.0']}]})
        self.store.register_image(STANDARD, b'standard preview', '/fixture/standard.jpg')
        request = copy.deepcopy(self.fixture.request)
        request['messages'][0]['content'] = STANDARD_SYSTEM
        self.fixture.add_record('raw_records', STANDARD, 'standard-log', 'inference_log', request)
        self.template = self.store.save_input(STANDARD, {'source_record_id': 'standard-log', 'reviewer': 'Fixture'})

    def tearDown(self):
        self.fixture.tearDown()

    def payload(self, fid=TARGET, **overrides):
        source = self.store.latest('input', fid)
        return {'adaptation_source': {key: source[key] for key in ('revision', 'input_hash')},
                'system_prompt_template': {key: self.template[key] for key in ('frame_id', 'revision', 'input_hash')},
                'expected_revision': source['revision'], 'reviewer': 'Fixture standardization', **overrides}

    def adapt(self, fid=TARGET, **overrides):
        return self.store.save_input(fid, self.payload(fid, **overrides))

    def test_logged_and_reconstructed_inputs_keep_all_frame_evidence_and_nested_provenance(self):
        for fid, provenance_key in ((TEMPLATE, 'source_request'), (TARGET, 'reconstruction')):
            with self.subTest(fid=fid):
                parent = self.store.latest('input', fid)
                saved = self.adapt(fid)
                self.assertEqual(saved['protocol_version'], ADAPTED_PROTOCOL)
                self.assertEqual(saved['revision'], parent['revision'] + 1)
                self.assertNotEqual(saved['input_hash'], parent['input_hash'])
                self.assertEqual(saved['system_prompt'], STANDARD_SYSTEM)
                self.assertEqual(saved['request_messages'][0], {'role': 'system', 'content': STANDARD_SYSTEM})
                self.assertEqual(saved['request_messages'][1], parent['request_messages'][1])
                for key in ('prompt', 'velocity', 'input_images', 'image_detail', 'image_layout', 'image_sha256', 'track'):
                    self.assertEqual(saved[key], parent[key])
                self.assertEqual(saved['adaptation']['parent_input'][provenance_key], parent[provenance_key])
                self.assertNotIn(provenance_key, saved)
                self.assertFalse(saved['adaptation']['historical_request_recovered'])
                self.assertEqual(saved['output_fields'], list(FIELDS))
                self.assertEqual(expected_output_fields(saved), FIELDS)
                self.assertEqual(archived_output_fields(saved), ('Description', 'Keywords'))
                self.assertEqual(saved['input_hash'], digest(input_semantics(saved)))
                self.assertEqual(verify_input_images(self.store, saved), saved['input_images'])
                self.assertIn(parent, self.store.versions('input', fid))
        self.assertEqual(self.store.latest('input', STANDARD), self.template)
        with self.assertRaisesRegex(ValueError, 'cannot overwrite a standardized input'):
            self.store.save_input(TEMPLATE, {'source_record_id': 'original-log', 'reviewer': 'Fixture', 'expected_revision': 2})

    def test_invalid_or_stale_sources_templates_and_overrides_cannot_create_a_version(self):
        baseline = self.store.versions('input', TARGET)
        variants = []
        bad = self.payload(); bad['adaptation_source']['input_hash'] = '0' * 64; variants.append(bad)
        bad = self.payload(); bad['system_prompt_template']['input_hash'] = '0' * 64; variants.append(bad)
        bad = self.payload(); bad['system_prompt_template'] = {key: self.fixture.template[key] for key in ('frame_id', 'revision', 'input_hash')}; variants.append(bad)
        variants += [self.payload(expected_revision=0), self.payload(source_record_id='fake'),
                     self.payload(prompt='new prompt'), self.payload(system_prompt='new system'),
                     self.payload(input_images=list(reversed(self.source['input_images'])))]
        for payload in variants:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError): self.store.save_input(TARGET, payload)
                self.assertEqual(self.store.versions('input', TARGET), baseline)
        stale = self.payload()
        self.adapt()
        with self.assertRaisesRegex(ValueError, 'current input revision'):
            self.store.save_input(TARGET, stale)

    def test_chat_and_responses_transport_exact_standard_system_and_original_user_evidence(self):
        self.adapt()
        job = build_job_preview('inference', [self.store.frame_detail(TARGET)])['jobs'][0]
        for api_mode in ('chat_completions', 'responses'):
            config = validate_config({'provider': 'openai', 'model': 'fixture', 'api_mode': api_mode})
            request, _, image_meta = _payload(job, config, self.store.root)
            self.assertEqual(len(image_meta), 2)
            self.assertNotIn('PRIVATE', json.dumps(request))
            if api_mode == 'chat_completions':
                expected = copy.deepcopy(self.fixture.manifest['request']['messages'])
                expected[0]['content'] = STANDARD_SYSTEM
                self.assertEqual(request['messages'], expected)
            else:
                self.assertEqual(request['input'][0]['content'][0]['text'], STANDARD_SYSTEM)
                self.assertEqual(request['input'][1]['content'][0]['text'], self.source['prompt'])
        gt = build_job_preview('ground_truth', [self.store.frame_detail(TARGET)], evidence_mode='sensory_only')['jobs'][0]
        self.assertNotIn('PRIVATE', json.dumps(gt))
        self.assertIn('assign tail facts to field Limb', gt['payload_template']['messages'][0]['content'])

    def test_modified_images_block_adaptation_and_approval_and_old_gt_is_stale(self):
        content = {'sections': {field: 'Fixture' for field in FIELDS}, 'facts': [
            {'id': 'f1', 'text': 'Tail up.', 'field': 'Limb', 'evidence': 'image', 'required': True}]}
        gt = self.store.save_gt(TARGET, {'status': 'approved', 'actor_type': 'human', 'human_confirmed': True,
            'approval_action': 'approve_reference', 'expected_revision': 0, 'expected_input_hash': self.source['input_hash'], 'content': content})
        saved = self.adapt()
        self.assertEqual(next(frame for frame in self.store.frame_list() if frame['id'] == TARGET)['gt_status'], 'needs_review')
        self.assertEqual(self.store.latest('gt', TARGET), gt)
        (self.store.root / saved['input_images'][1]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.adapt()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.store.save_gt(TARGET, {'status': 'approved', 'actor_type': 'human', 'human_confirmed': True,
                'approval_action': 'approve_reference', 'expected_revision': gt['revision'],
                'expected_input_hash': saved['input_hash'], 'content': content})

    def test_historical_outputs_keep_old_format_and_cannot_claim_adapted_input_match(self):
        output = '{"Description":"Fixture behavior.","Keywords":"fixture"}'
        asset = {'id': 'old-output', 'frame_id': TARGET, 'model_label': 'old model', 'nas_path': '/fixture/output.json'}
        self.store.register_text(asset, output, hashlib.sha256(output.encode()).hexdigest())
        rating = {'id': 'old-rating', 'frame_id': TARGET, 'kind': 'single', 'model_label': 'old model'}
        with self.store.transaction():
            self.store.db.execute('INSERT INTO records VALUES(?,?,?,?)', ('ratings', rating['id'], TARGET, dumps(rating)))
        run = self.store.import_run({'name': 'Old output', 'model': 'old model', 'records': [
            {'frame_id': TARGET, 'output': output, 'input_hash': self.source['input_hash']}]})
        saved = self.adapt()
        detail = self.store.frame_detail(TARGET)
        self.assertEqual(detail['output_format']['expected_fields'], list(FIELDS))
        self.assertTrue(detail['descriptions'][0]['output_view']['contract_valid'])
        self.assertTrue(self.store.predictions(run['id'])[0]['mechanical_checks']['passed'])
        workflow = Workflow(self.store)
        with self.assertRaisesRegex(ValueError, 'older or unknown request'):
            workflow.verify_evidence_link({'frame_id': TARGET, 'rating_ids': ['old-rating'], 'run_id': run['id'],
                'reviewer': 'Fixture', 'reason': 'Must reject request mismatch', 'human_confirmed': True, 'input_verified': True})
        # Even a forged portable link cannot mark the known old request as current.
        detail['ground_truth'] = {'status': 'approved', 'revision': 2, 'content': {'facts': []},
                                  'content_sha256': 'a' * 64, 'input_hash': saved['input_hash']}
        detail['prediction'] = self.store.predictions(run['id'])[0]
        detail['input_evidence_links'] = [{'frame_id': TARGET, 'input_verified': True, 'human_confirmed': True,
            'input_sha256': saved['input_hash'], 'prediction_hashes': [detail['prediction']['prediction_sha256']]}]
        job = build_job_preview('judge', [detail], run=run)['jobs'][0]
        candidate = json.loads(job['payload_template']['messages'][1]['content'])['candidate_output']
        self.assertFalse(candidate['frozen_request_match_verified'])
        self.assertTrue(candidate['known_request_mismatch'])
        self.assertTrue(candidate['output_view']['contract_valid'])

    def test_old_pending_jobs_cannot_be_claimed_after_input_standardization(self):
        workflow = Workflow(self.store)
        batch = workflow.create_batch({'task': 'ground_truth', 'executor': 'codex', 'model': 'fixture', 'frame_ids': [TARGET]})
        self.adapt()
        with self.assertRaisesRegex(ValueError, 'Input changed after this job'):
            workflow.claim(batch['batch']['id'], 'Fixture worker')
        job = workflow.job(batch['jobs'][0]['id'])
        self.assertEqual(job['status'], 'pending')
        self.assertEqual(job['attempts'], 0)


if __name__ == '__main__': unittest.main()
