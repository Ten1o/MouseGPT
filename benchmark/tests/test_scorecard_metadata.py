import copy
import json
import tempfile
import unittest

from mousebench.store import FIELDS, Store, reasoning_effort_metadata


class ScorecardMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.frame = '240420_C10M1_00003972'
        self.store.import_catalog({'frames': [{'id': self.frame, 'recording_id': 'fixture'}]})
        self.store.register_image(self.frame, b'fixture-image', 'fixture')
        self.contract = self.store.save_input(self.frame, {'prompt': 'Fixture prompt', 'reviewer': 'Fixture'})
        content = {'reference_description': 'Fixture reference',
                   'facts': [{'id': 'f1', 'text': 'Fixture fact', 'field': 'Overall', 'evidence': 'image', 'required': True}],
                   'uncertainties': [], 'sections': {field: 'Fixture' for field in FIELDS}}
        self.store.save_gt(self.frame, {'status': 'reviewed', 'content': content, 'reviewer': 'Fixture'})
        self.gt = self.store.save_gt(self.frame, {'status': 'approved', 'content': content,
            'reviewer': 'Fixture', 'human_confirmed': True, 'expected_revision': 1})
        self.primary_run = self.make_run('Misleading name effort=max', {'reasoning_effort': 'low'})
        self.other_run = self.make_run('No requested effort', {})
        for version, run in [('old', self.primary_run), ('middle', self.primary_run), ('new', self.primary_run), ('new', self.other_run)]:
            prediction = self.store.predictions(run['id'])[0]
            self.store.import_judgments({'records': [{
                'schema_version': 'mousebench.judgment.v1', 'frame_id': self.frame, 'run_id': run['id'],
                'protocol_version': 'fixture-v1', 'judge': {'provider': 'fixture', 'model': 'fixture', 'version': version},
                'rubric_sha256': 'a' * 64, 'input_sha256': self.contract['input_hash'],
                'ground_truth_sha256': self.gt['content_sha256'], 'prediction_sha256': prediction['prediction_sha256'],
                'claims': [{'text': 'Fixture fact', 'label': 'supported'}], 'supported_fact_ids': ['f1'],
                'structure_score': 1, 'instruction_score': 1}]})

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def make_run(self, name, settings):
        return self.store.import_run({'name': name, 'model': 'same-fixture-model', 'records': [{
            'frame_id': self.frame, 'output': 'Fixture output', 'model_metadata': {'generation_settings': settings}}]})

    def snapshot(self):
        snapshot = self.store.scorecard_snapshot()
        dates = {'old': '2026-10-01T01:00:00+00:00', 'middle': '2026-10-02T01:00:00+00:00',
                 'new': '2026-10-03T01:00:00+00:00'}
        for entry in snapshot['entries']:
            entry['created_at'] = dates[entry['judge']['version']]
            if entry['run_id'] == self.other_run['id']:
                entry['created_at'] = '2026-10-04T01:00:00+00:00'
        # Receipt iteration order need not match chronology.
        snapshot['entries'].reverse()
        return snapshot

    def test_newest_first_with_chronological_numbers_and_timestamp_bounds(self):
        snapshot = self.snapshot()
        before = copy.deepcopy(snapshot)
        stored_before = list(self.store.db.iterdump())
        cards = self.store.scorecards(snapshot=snapshot)
        self.assertEqual([g['judge']['version'] for g in cards['groups']], ['new', 'middle', 'old'])
        self.assertEqual([g['comparison_number'] for g in cards['groups']], [3, 2, 1])
        self.assertEqual(cards['groups'][0]['created_at'], '2026-10-03T01:00:00+00:00')
        self.assertEqual(cards['groups'][0]['updated_at'], '2026-10-04T01:00:00+00:00')
        order = {group['group_id']: group['comparison_number'] for group in cards['groups']}
        self.assertEqual([order[m['group_id']] for m in cards['models']], [3, 3, 2, 1])
        for model in cards['models']:
            self.assertEqual(model['dimension_means']['factual_f1'], 1)
            self.assertEqual(model['evaluable_count'], 1)
        self.assertEqual(snapshot, before)
        self.assertEqual(list(self.store.db.iterdump()), stored_before)

    def test_missing_or_invalid_timestamps_stay_unknown_and_do_not_fail(self):
        snapshot = self.snapshot()
        for entry in snapshot['entries']:
            if entry['judge']['version'] == 'old':
                entry.pop('created_at')
            if entry['judge']['version'] == 'middle':
                entry['created_at'] = 'invalid-date'
        cards = self.store.scorecards(snapshot=snapshot)
        self.assertEqual(cards['groups'][0]['judge']['version'], 'new')
        for group in cards['groups'][1:]:
            self.assertIsNone(group['created_at'])
            self.assertIsNone(group['updated_at'])

    def test_ineligible_group_does_not_renumber_later_groups(self):
        snapshot = self.snapshot()
        for entry in snapshot['entries']:
            if entry['judge']['version'] == 'middle':
                entry['ground_truth_sha256'] = 'stale'
        cards = self.store.scorecards(snapshot=snapshot)
        self.assertEqual([g['comparison_number'] for g in cards['groups']], [3, 1])
        self.assertEqual(cards['stale_judgments_excluded'], 1)

    def test_effort_comes_from_recorded_settings_not_run_name(self):
        cards = self.store.scorecards(snapshot=self.snapshot())
        by_run = {model['run_id']: model for model in cards['models']}
        self.assertEqual(by_run[self.primary_run['id']]['reasoning_effort'], 'low')
        self.assertEqual(by_run[self.primary_run['id']]['reasoning_effort_status'], 'explicit')
        self.assertIsNone(by_run[self.other_run['id']]['reasoning_effort'])
        self.assertEqual(by_run[self.other_run['id']]['reasoning_effort_status'], 'not_requested')
        self.assertEqual(reasoning_effort_metadata([None]), {'reasoning_effort': None, 'reasoning_effort_status': 'unrecorded'})
        self.assertEqual(reasoning_effort_metadata([{'reasoning_effort': 'none'}]),
            {'reasoning_effort': 'none', 'reasoning_effort_status': 'explicit'})
        self.assertEqual(reasoning_effort_metadata([{'reasoning_effort': 'low'}, {'reasoning_effort': 'high'}]),
            {'reasoning_effort': None, 'reasoning_effort_status': 'mixed'})

    def test_judge_effort_requires_matching_frozen_config_model_and_provider(self):
        snapshot = self.snapshot()
        for entry in snapshot['entries']:
            if entry['judge']['version'] == 'new':
                entry['judge']['version'] = 'fixture|config:exact-sha'
        self.store.db.execute('CREATE TABLE wf_records(kind TEXT,id TEXT,data TEXT)')
        for ident, model, effort in [('match', 'fixture', 'medium'), ('other-model', 'other', 'high')]:
            batch = {'task': 'judge', 'execution_config_sha256': 'exact-sha', 'executor': 'fixture',
                     'model': model, 'provider_config': {'settings': {'reasoning_effort': effort}}}
            self.store.db.execute('INSERT INTO wf_records VALUES(?,?,?)', ('batch', ident, json.dumps(batch)))
        cards = self.store.scorecards(snapshot=snapshot)
        matching = next(group for group in cards['groups'] if '|config:' in group['judge']['version'])
        self.assertEqual(matching['judge_reasoning_effort'], 'medium')
        self.assertEqual(matching['judge_reasoning_effort_status'], 'explicit')
        for group in cards['groups']:
            if group is not matching:
                self.assertEqual(group['judge_reasoning_effort_status'], 'unrecorded')


if __name__ == '__main__':
    unittest.main()
