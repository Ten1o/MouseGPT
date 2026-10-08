import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mousebench.recorded_inputs import recorded_input, input_semantics, verify_input_images, PROTOCOL
from mousebench.store import Store, digest, dumps, FIELDS


FRAME = '240419_C11M4_00003972'
OTHER = '240419_C11M4_00003973'
PROMPT = ('  The first two photos shot at different view at the same time. '
          'The overall movement speed of the mouse is: 9.77 cm/s, the speed of the head: 40.60 cm/s, '
          'speed of left and right forelimbs: 22.47, 9.76 cm/s, and left and right hind limbs are '
          '10.09, 10.33 cm/s respectively.\n')


class RecordedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.store.import_catalog({'frames': [{'id': FRAME}, {'id': OTHER}]})
        self.store.register_image(FRAME, b'fixture concatenated image', '/fixture/concat.jpg')
        self.images = [b'\xff\xd8\xfffixture-first-view\xff\xd9', b'\xff\xd8\xfffixture-second-view\xff\xd9']
        self.source = {'messages': [
            {'role': 'system', 'content': '  Original system instructions.\n'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': PROMPT},
                *[{'type': 'image_url', 'image_url': {
                    'url': 'data:image/jpeg;base64,' + base64.b64encode(blob).decode(), 'detail': 'high'}}
                  for blob in self.images]]}],
            'top_p': 0.5, 'temperature': 0.3, 'max_tokens': 2000,
            'response': {'secret_output': 'Never use an output as input.'}}
        self.record = self.add_source('log-1', self.source)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def add_source(self, record_id, source, frame=FRAME, source_path=None):
        raw = json.dumps(source, ensure_ascii=False).encode()
        sha = hashlib.sha256(raw).hexdigest()
        path = 'objects/' + sha + '.json'
        (self.store.root / path).write_bytes(raw)
        record = {'id': record_id, 'frame_id': frame, 'frame_membership_verified': True,
                  'role': 'inference_log', 'path': path, 'sha256': sha,
                  'nas_path': source_path or '/fixture/' + frame + '/log.json'}
        self.set_record(record)
        return record

    def set_record(self, record, db_frame=None):
        with self.store.transaction():
            self.store.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',
                                  ('raw_records', record['id'], db_frame or record['frame_id'], dumps(record)))

    def save(self, **extra):
        return self.store.save_input(FRAME, {'source_record_id': 'log-1', 'reviewer': 'Codex archive importer',
                                            'expected_revision': 0, **extra})

    def test_read_only_preview_preserves_exact_request_and_never_exposes_outputs(self):
        files = sorted(path.name for path in (self.store.root / 'objects').iterdir())
        before = list(self.store.db.iterdump())
        preview = self.store.frame_detail(FRAME)['recorded_input']
        self.assertEqual(preview['status'], 'ready')
        self.assertEqual(preview['prompt'], PROMPT)
        self.assertEqual(preview['system_prompt'], '  Original system instructions.\n')
        self.assertEqual(preview['velocity']['value'], {'overall': 9.77, 'head': 40.60,
                         'left_forelimb': 22.47, 'right_forelimb': 9.76, 'left_hind_limb': 10.09, 'right_hind_limb': 10.33})
        self.assertEqual(preview['velocity']['units'], 'cm/s')
        self.assertTrue(preview['velocity_embedded_in_prompt'])
        self.assertEqual(preview['generation_settings'], {'top_p': 0.5, 'temperature': 0.3, 'max_tokens': 2000})
        self.assertNotIn('secret_output', json.dumps(preview))
        self.assertEqual(files, sorted(path.name for path in (self.store.root / 'objects').iterdir()))
        self.assertEqual(before, list(self.store.db.iterdump()))

    def test_saved_contract_preserves_text_image_order_and_source_provenance(self):
        saved = self.save()
        self.assertEqual(saved['protocol_version'], PROTOCOL)
        self.assertEqual(saved['prompt'], PROMPT)
        self.assertEqual(saved['system_prompt'], self.source['messages'][0]['content'])
        self.assertEqual(saved['image_layout'], 'historical_separate_views')
        self.assertEqual(saved['track'], 'image_velocity')
        self.assertEqual(saved['image_sha256'], self.store.image(FRAME)['sha256'])
        self.assertEqual(saved['source_request'], {'record_id': 'log-1', 'source_path': self.record['nas_path'],
                         'sha256': self.record['sha256'],
                         'generation_settings': {'top_p': 0.5, 'temperature': 0.3, 'max_tokens': 2000}})
        for index, (image, blob) in enumerate(zip(saved['input_images'], self.images), 1):
            self.assertEqual(image['attachment_id'], 'input_image_' + str(index))
            self.assertEqual((self.store.root / image['path']).read_bytes(), blob)
            self.assertEqual(image['sha256'], hashlib.sha256(blob).hexdigest())
            self.assertEqual(image['size'], len(blob))
        self.assertEqual(saved['request_messages'][1]['content'], [
            {'type': 'text', 'text': PROMPT}, {'type': 'image_reference', 'attachment_id': 'input_image_1'},
            {'type': 'image_reference', 'attachment_id': 'input_image_2'}])
        self.assertEqual(saved['input_hash'], digest(input_semantics(saved)))

    def test_existing_left_first_contract_retains_prechange_semantic_hash(self):
        # Captured from the established v1.0 parser before adding older side orders.
        self.assertEqual(self.save()['input_hash'],
                         '136dfdab9f694215f529084481dfa261ec575eae4930b4b33d8b9e1b714f55ec')

    def test_limb_speed_order_is_read_independently_without_rewriting_prompt_or_images(self):
        for fore_order in ('left and right', 'right and left'):
            for hind_order in ('left and right', 'right and left'):
                with self.subTest(fore=fore_order, hind=hind_order):
                    prompt = PROMPT.replace('left and right forelimbs', fore_order+' forelimbs')
                    prompt = prompt.replace('left and right hind limbs', hind_order+' hind limbs')
                    source = copy.deepcopy(self.source)
                    source['messages'][1]['content'][0]['text'] = prompt
                    self.add_source('log-1', source)
                    preview = recorded_input(self.store, FRAME)
                    self.assertEqual(preview['status'], 'ready')
                    self.assertEqual(preview['prompt'], prompt)
                    self.assertEqual(preview['request_messages'][1]['content'][0]['text'], prompt)
                    speeds = preview['velocity']['value']
                    first_fore, second_fore = fore_order.split(' and ')
                    first_hind, second_hind = hind_order.split(' and ')
                    self.assertEqual(speeds[first_fore+'_forelimb'], 22.47)
                    self.assertEqual(speeds[second_fore+'_forelimb'], 9.76)
                    self.assertEqual(speeds[first_hind+'_hind_limb'], 10.09)
                    self.assertEqual(speeds[second_hind+'_hind_limb'], 10.33)
                    self.assertEqual([image['sha256'] for image in preview['input_images']],
                                     [hashlib.sha256(blob).hexdigest() for blob in self.images])

    def test_repeated_or_ambiguous_limb_side_labels_are_rejected(self):
        bad_prompts = [PROMPT.replace('left and right forelimbs', order+' forelimbs')
                       for order in ('left and left', 'right and right', 'front and back', 'left/right')]
        bad_prompts += [PROMPT.replace('left and right hind limbs', order+' hind limbs')
                        for order in ('left and left', 'right and right', 'right and unknown')]
        bad_prompts.append(PROMPT + PROMPT)
        for prompt in bad_prompts:
            with self.subTest(prompt=prompt):
                source = copy.deepcopy(self.source)
                source['messages'][1]['content'][0]['text'] = prompt
                self.add_source('log-1', source)
                self.assertEqual(recorded_input(self.store, FRAME)['status'], 'error')
                with self.assertRaises(ValueError):
                    self.save()
        self.assertIsNone(self.store.latest('input', FRAME))

    def test_seed_is_preserved_as_source_setting_and_conflicting_copies_are_rejected(self):
        source = copy.deepcopy(self.source)
        source['seed'] = 42
        self.add_source('log-1', source)
        saved = self.save()
        self.assertEqual(saved['source_request']['generation_settings']['seed'], 42)
        self.assertEqual(saved['prompt'], PROMPT)
        # Archived generation settings remain source provenance, preserving the
        # established request hash used by existing controlled-input contracts.
        self.assertEqual(saved['input_hash'], '136dfdab9f694215f529084481dfa261ec575eae4930b4b33d8b9e1b714f55ec')
        changed = copy.deepcopy(source)
        changed['seed'] = 43
        self.add_source('log-copy', changed)
        self.assertEqual(recorded_input(self.store, FRAME)['status'], 'conflict')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.save(expected_revision=1)
        self.assertEqual(len(self.store.versions('input', FRAME)), 1)

    def test_invalid_seed_is_rejected_and_explicit_null_is_preserved(self):
        for seed in (True, '42', 42.5):
            with self.subTest(seed=seed):
                source = copy.deepcopy(self.source)
                source['seed'] = seed
                self.add_source('log-1', source)
                self.assertEqual(recorded_input(self.store, FRAME)['status'], 'error')
        source['seed'] = None
        self.add_source('log-1', source)
        self.assertIsNone(recorded_input(self.store, FRAME)['generation_settings']['seed'])

    def test_hash_ignores_source_locations_reviewer_time_but_binds_bytes_order_and_text(self):
        first = self.save()
        second = self.save(expected_revision=1, reviewer='Different importer')
        self.assertEqual(first['input_hash'], second['input_hash'])
        relocated = copy.deepcopy(first)
        relocated['source_request']['source_path'] = '/another/source'
        relocated['velocity']['source'] = '/another/source'
        relocated['input_images'][0]['path'] = 'objects/another-location.jpg'
        self.assertEqual(digest(input_semantics(relocated)), first['input_hash'])
        for mutate in (lambda value: value.update(prompt=value['prompt'] + ' '),
                       lambda value: value['input_images'].reverse(),
                       lambda value: value['input_images'][0].update(sha256='a' * 64),
                       lambda value: value['request_messages'][1]['content'].reverse()):
            changed = copy.deepcopy(first)
            mutate(changed)
            self.assertNotEqual(digest(input_semantics(changed)), first['input_hash'])

    def test_edited_historical_fields_are_rejected_without_writing_contract(self):
        for changes in ({'prompt': PROMPT.strip()}, {'system_prompt': 'replacement'},
                        {'velocity': None}, {'image_detail': 'low'}, {'input_images': []}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'does not match'):
                self.save(**changes)
        self.assertIsNone(self.store.latest('input', FRAME))

    def test_tampered_source_and_path_escape_are_blocked(self):
        path = self.store.root / self.record['path']
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        self.assertEqual(recorded_input(self.store, FRAME)['status'], 'error')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.save()
        path.write_bytes(original)
        self.set_record(dict(self.record, path='../outside.json'))
        with self.assertRaisesRegex(ValueError, 'inside'):
            self.save()

    def test_frame_mismatch_or_unverified_membership_blocked(self):
        self.add_source('other-log', self.source, frame=OTHER)
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            self.save(source_record_id='other-log')
        self.set_record(dict(self.record, frame_id=OTHER), db_frame=FRAME)
        with self.assertRaisesRegex(ValueError, 'membership'):
            self.save()
        self.set_record(dict(self.record, frame_membership_verified=False))
        with self.assertRaisesRegex(ValueError, 'membership'):
            self.save()

    def test_conflicting_requests_block_even_when_id_selected(self):
        changed = copy.deepcopy(self.source)
        changed['messages'][1]['content'][0]['text'] = PROMPT.replace('9.77', '9.78')
        self.add_source('log-2', changed)
        self.assertEqual(recorded_input(self.store, FRAME)['status'], 'conflict')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.save()

    def test_identical_copies_resolve_and_bind_selected_provenance(self):
        self.add_source('log-copy', self.source, source_path='/fixture/copy/log.json')
        self.assertEqual(recorded_input(self.store, FRAME)['status'], 'ready')
        saved = self.save(source_record_id='log-copy')
        self.assertEqual(saved['source_request']['record_id'], 'log-copy')
        self.assertEqual(saved['source_request']['source_path'], '/fixture/copy/log.json')

    def test_malformed_images_speeds_and_assistant_messages_do_not_fallback(self):
        broken = []
        value = copy.deepcopy(self.source)
        value['messages'][1]['content'][1]['image_url']['url'] = 'https://example.test/image.jpg'
        broken.append(value)
        value = copy.deepcopy(self.source)
        value['messages'][1]['content'][1]['image_url']['url'] = 'data:image/jpeg;base64,!!!'
        broken.append(value)
        value = copy.deepcopy(self.source)
        value['messages'][1]['content'][0]['text'] = 'A prompt without verified speeds.'
        broken.append(value)
        value = copy.deepcopy(self.source)
        value['messages'].append({'role': 'assistant', 'content': 'An output.'})
        broken.append(value)
        for value in broken:
            with self.subTest(source=value):
                self.add_source('log-1', value)
                self.assertEqual(recorded_input(self.store, FRAME)['status'], 'error')
                with self.assertRaises(ValueError):
                    self.save()
        self.assertIsNone(self.store.latest('input', FRAME))

    def test_existing_corrupt_object_is_not_overwritten(self):
        preview = recorded_input(self.store, FRAME)
        path = self.store.root / preview['input_images'][0]['path']
        path.write_bytes(b'corrupt existing file')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.save()
        self.assertEqual(path.read_bytes(), b'corrupt existing file')
        self.assertIsNone(self.store.latest('input', FRAME))

    def test_unavailable_source_is_explicit_and_manual_v2_still_works(self):
        self.assertEqual(recorded_input(self.store, OTHER)['status'], 'unavailable')
        saved = self.store.save_input(FRAME, {'prompt': '  Manual fixture. ', 'reviewer': 'test'})
        self.assertEqual(saved['protocol_version'], 'mousebench.input.v2')
        self.assertEqual(saved['input_hash'], digest(input_semantics(saved)))
        self.assertEqual(saved['prompt'], 'Manual fixture.')

    def test_review_requires_intact_original_views_not_only_concat_preview(self):
        contract = self.save()
        self.assertEqual(verify_input_images(self.store, contract), contract['input_images'])
        path = self.store.root / contract['input_images'][0]['path']
        path.unlink()
        content = {'facts': [{'id': 'f1', 'text': 'Fixture fact.', 'field': 'Overall',
                             'evidence': 'image', 'required': True}],
                   'sections': {field: 'Fixture only.' for field in FIELDS}, 'uncertainties': []}
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.store.save_gt(FRAME, {'status': 'reviewed', 'reviewer': 'Fixture reviewer', 'content': content})
        self.assertIsNotNone(self.store.image(FRAME))
        self.assertIsNone(self.store.latest('gt', FRAME))

    def test_integrity_helper_rejects_contract_tamper_and_path_escape(self):
        saved = self.save()
        changed = copy.deepcopy(saved)
        changed['prompt'] += 'tamper'
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_input_images(self.store, changed)
        changed = copy.deepcopy(saved)
        changed['input_images'][0]['path'] = '../outside.jpg'
        with self.assertRaisesRegex(ValueError, 'outside'):
            verify_input_images(self.store, changed)


if __name__ == '__main__':
    unittest.main()
