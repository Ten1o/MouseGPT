"""Offline, version-bound read-only scoring inspection."""
import copy
import json
import unittest
from unittest.mock import patch

import test_graded_workflow as graded
import test_workflow as fixtures
from mousebench.scoring_review import build_review, review_detail, review_image, prediction_detail, prediction_image, _presentation_view
from mousebench.store import dumps


class ScoringReviewTests(unittest.TestCase):
    setUp = fixtures.WorkflowTests.setUp
    tearDown = fixtures.WorkflowTests.tearDown
    inputs = fixtures.WorkflowTests.inputs
    batch = fixtures.WorkflowTests.batch
    receipt = fixtures.WorkflowTests.receipt
    approve = fixtures.WorkflowTests.approve
    candidate = fixtures.WorkflowTests.candidate
    prepare = graded.GradedWorkflowTests.prepare
    judgment = graded.GradedWorkflowTests.judgment

    def ready(self):
        rubric, run, job = self.prepare()
        self.workflow.complete(self.receipt(job, self.judgment()))
        return rubric, run, self.store.export('judgments')[0]

    def test_no_runs_and_unjudged_predictions_have_no_fake_scores(self):
        empty = build_review(self.store)
        self.assertEqual(empty['frames'], [])
        self.assertIsNone(empty['selected_run_id'])
        self.inputs()
        run = self.candidate(frames=fixtures.FRAMES)
        result = build_review(self.store, run)
        self.assertEqual(result['counts']['unjudged'], 3)
        self.assertEqual(result['groups'], [])
        self.assertTrue(all(row['scores'] is None and row['judgment_id'] is None for row in result['frames']))
        detail = prediction_detail(self.store, run, fixtures.FRAMES[0])
        self.assertIsNone(detail['ground_truth'])
        self.assertIsNone(detail['judgment'])
        self.assertIsNone(detail['scores'])
        self.assertEqual(detail['prediction']['text'], detail['prediction']['output'])
        self.assertTrue(detail['images'])
        blob, mime = prediction_image(self.store, run, fixtures.FRAMES[0], 'canonical')
        self.assertIn(b'offline-fixture', blob)
        self.assertEqual(mime, 'image/jpeg')

    def test_scores_match_canonical_scorecard_and_reads_do_not_mutate(self):
        rubric, run, j = self.ready()
        before = self.store.db.total_changes
        before_judgments = self.store.export('judgments')
        result = build_review(self.store, run)
        row = result['frames'][0]
        card = self.workflow.scorecards()['models'][0]
        self.assertEqual(row['status'], 'judged')
        self.assertEqual(row['benchmark_index'], card['benchmark_index'])
        self.assertEqual(result['selected_group_id'], card['group_id'])
        detail = review_detail(self.store, j['id'])
        self.assertEqual(detail['scores']['factual_f1'], card['dimension_means']['factual_f1'])
        self.assertEqual(detail['claims'][0]['weight'], 1)
        self.assertEqual(detail['claims'][0]['weighted_correctness_credit'], 1)
        self.assertEqual(detail['claims'][0]['facts'][0]['text'], self.content['facts'][0]['text'])
        self.assertEqual(detail['fact_coverage'][0]['weighted_coverage_credit'], .75)
        self.assertEqual(detail['ground_truth']['revision'], j['gt_revision'])
        self.assertEqual(detail['index_details']['value'], card['benchmark_index'])
        self.assertEqual(detail['rubric']['sha256'], rubric['sha256'])
        self.assertEqual(len(detail['criteria']), 8)
        self.assertEqual(self.store.db.total_changes, before)
        self.assertEqual(self.store.export('judgments'), before_judgments)

    def test_stale_reference_uses_exact_old_fact_text_and_scores(self):
        _, run, j = self.ready()
        before = review_detail(self.store, j['id'])
        latest = self.store.latest('gt', j['frame_id'])
        content = copy.deepcopy(latest['content'])
        content['facts'][0]['text'] = 'Different revised direction.'
        self.store.save_gt(j['frame_id'], {'content': content, 'status': 'draft',
            'reviewer': 'Fixture', 'expected_revision': latest['revision']})
        result = build_review(self.store, run)
        self.assertEqual(result['counts']['stale'], 1)
        detail = review_detail(self.store, j['id'])
        self.assertFalse(detail['eligibility']['eligible'])
        self.assertEqual(detail['ground_truth'], before['ground_truth'])
        self.assertEqual(detail['scores'], before['scores'])
        self.assertEqual(detail['claims'][0]['facts'][0]['text'], self.content['facts'][0]['text'])
        self.assertEqual(self.workflow.scorecards()['stale_judgments_excluded'], 1)

    def test_stale_input_image_never_uses_newest_image_or_prompt(self):
        _, run, j = self.ready()
        before = review_detail(self.store, j['id'])
        old_bytes, _ = review_image(self.store, j['id'], 'canonical')
        contract = self.store.latest('input', j['frame_id'])
        self.store.register_image(j['frame_id'], b'new image content', '/fixture/new.jpg')
        self.store.save_input(j['frame_id'], {'prompt': 'A changed input prompt.',
            'expected_revision': contract['revision'], 'reviewer': 'Fixture'})
        detail = review_detail(self.store, j['id'])
        self.assertEqual(detail['eligibility']['status'], 'stale')
        self.assertEqual(detail['input'], before['input'])
        self.assertEqual(detail['images'], before['images'])
        self.assertEqual(review_image(self.store, j['id'], 'canonical')[0], old_bytes)
        self.assertEqual(build_review(self.store, run)['counts']['stale'], 1)
        old_path = self.store.root / detail['images'][0]['path']
        old_path.write_bytes(b'corrupt old evidence')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            review_image(self.store, j['id'], 'canonical')
        self.assertFalse(review_detail(self.store, j['id'])['images'][0]['available'])

    def test_missing_bound_reference_is_not_replaced(self):
        _, _, j = self.ready()
        self.store.db.execute("DELETE FROM versions WHERE kind='gt' AND frame_id=?", (j['frame_id'],))
        detail = review_detail(self.store, j['id'])
        self.assertIsNone(detail['ground_truth'])
        self.assertIsNone(detail['scores'])
        self.assertIsNone(detail['benchmark_index'])
        self.assertEqual(detail['judgment']['scores'], j['scores'])
        self.assertTrue(any('bound ground-truth' in warning for warning in detail['warnings']))
        self.assertEqual(detail['eligibility']['status'], 'ineligible')

    def test_missing_bound_rubric_never_uses_default_policy(self):
        rubric, run, j = self.ready()
        self.store.db.execute("DELETE FROM wf_records WHERE kind='rubric' AND id=?", (rubric['id'],))
        detail = review_detail(self.store, j['id'])
        self.assertIsNone(detail['scores'])
        self.assertIsNone(detail['benchmark_index'])
        self.assertEqual(detail['eligibility']['status'], 'ineligible')
        self.assertEqual(build_review(self.store, run)['counts']['ineligible'], 1)

    def test_code_fence_is_only_readable_display_not_format_repair(self):
        output = '```json\n{"Overall":"Unchanged answer."}\n```'
        run = self.store.import_run({'name': 'Archive', 'model': 'Archive', 'track': 'historical',
            'records': [{'frame_id': fixtures.FRAMES[0], 'output': output}]})
        detail = prediction_detail(self.store, run['id'], fixtures.FRAMES[0])
        prediction = detail['prediction']
        self.assertEqual(prediction['text'], output)
        self.assertFalse(prediction['output_view']['json_valid'])
        self.assertTrue(prediction['presentation_view']['json_valid'])
        self.assertEqual(prediction['presentation_view']['fields'][0]['value'], 'Unchanged answer.')
        for invalid in ('prefix\n' + output, output + '\nsuffix', output + '\n' + output,
                        '```json\n{bad json}\n```', '```python\n{}\n```'):
            self.assertIsNone(_presentation_view(invalid, ('Overall',)))
        duplicate = _presentation_view('```json\n{"Overall":"a","Overall":"b"}\n```', ('Overall',))
        self.assertEqual(duplicate['fields'][0]['status'], 'duplicate')
        self.assertIsNone(duplicate['fields'][0]['value'])

    def test_missing_or_mismatched_model_answer_is_not_substituted(self):
        _, _, j = self.ready()
        row = self.store.db.execute('SELECT data FROM predictions WHERE run_id=? AND frame_id=?',
            (j['run_id'], j['frame_id'])).fetchone()
        prediction = json.loads(row[0])
        prediction['output'] = 'Tampered answer with unchanged stored hash.'
        self.store.db.execute('UPDATE predictions SET data=? WHERE run_id=? AND frame_id=?',
            (dumps(prediction), j['run_id'], j['frame_id']))
        detail = review_detail(self.store, j['id'])
        self.assertIsNone(detail['prediction'])
        self.assertTrue(any('exact model answer' in warning for warning in detail['warnings']))

    def test_judge_rubric_and_scoring_versions_are_separate_groups(self):
        _, run, j = self.ready()
        altered = copy.deepcopy(j)
        altered['id'] = 'other-judge'
        altered['judge']['version'] = 'independent-v2'
        self.store.db.execute('INSERT INTO judgments VALUES(?,?,?,?)', (altered['id'], run, altered['frame_id'], dumps(altered)))
        review = build_review(self.store, run)
        self.assertEqual(len(review['groups']), 2)
        self.assertEqual(len(set(g['group_id'] for g in review['groups'])), 2)
        for group in review['groups']:
            selected = build_review(self.store, run, group['group_id'])
            self.assertEqual(selected['counts']['judgments'], 1)
            detail = review_detail(self.store, selected['frames'][0]['judgment_id'])
            self.assertEqual(detail['group']['judge'], group['judge'])
        altered['id'] = 'other-rubric'
        altered['rubric_sha256'] = 'e' * 64
        self.store.db.execute('INSERT INTO judgments VALUES(?,?,?,?)', (altered['id'], run, altered['frame_id'], dumps(altered)))
        self.assertEqual(len(build_review(self.store, run)['groups']), 3)

    def test_critical_issue_eligibility_matches_scorecards(self):
        _, run, j = self.ready()
        self.workflow.put('issue', {'id': 'blocking-fixture', 'frame_id': j['frame_id'], 'status': 'open',
            'severity': 'critical', 'target': 'reference', 'content_sha256': j['ground_truth_sha256'],
            'input_sha256': j['input_sha256']})
        review = build_review(self.store, run)
        self.assertEqual(review['counts']['ineligible'], 1)
        self.assertEqual(self.workflow.scorecards()['critical_issue_judgments_excluded'], 1)
        detail = review_detail(self.store, j['id'])
        self.assertEqual(detail['eligibility']['status'], 'ineligible')
        self.assertIsNotNone(detail['scores'])

    def test_list_batches_issue_reads_without_caching_across_requests(self):
        _, run, j = self.ready()
        # Several retained revisions still need only one current issue lookup.
        for suffix in ('second', 'third'):
            retained = dict(j, id='retained-' + suffix)
            self.store.db.execute('INSERT INTO judgments VALUES(?,?,?,?)',
                (retained['id'], run, retained['frame_id'], dumps(retained)))
        with patch.object(self.store, 'critical_issue_bindings',
                          wraps=self.store.critical_issue_bindings) as bindings, \
                patch.object(self.store, 'has_critical_issue',
                             side_effect=AssertionError('Repeated issue scan')), \
                patch.object(self.store, 'predictions',
                             side_effect=AssertionError('List must not reparse complete answers')):
            review = build_review(self.store, run)
            self.assertEqual(review['counts']['judgments'], 3)
            self.assertEqual(review['counts']['judged'], 1)
            self.assertEqual(bindings.call_count, 1)
        issue = {'id': 'new-critical-issue', 'frame_id': j['frame_id'], 'status': 'open',
                 'severity': 'critical', 'target': 'reference',
                 'content_sha256': j['ground_truth_sha256'], 'input_sha256': j['input_sha256']}
        self.workflow.put('issue', issue)
        self.assertEqual(build_review(self.store, run)['counts']['ineligible'], 1)
        self.workflow.put('issue', dict(issue, status='resolved'))
        self.assertEqual(build_review(self.store, run)['counts']['judged'], 1)

    def test_historical_unjudged_keeps_unknown_input_unknown(self):
        self.inputs()
        run = self.store.import_run({'name': 'Archive', 'model': 'Unknown archive', 'track': 'historical',
            'records': [{'frame_id': fixtures.FRAMES[0], 'output': '{"Overall":"Existing output"}'}]})
        detail = prediction_detail(self.store, run['id'], fixtures.FRAMES[0])
        self.assertIsNone(detail['input'])
        self.assertIsNone(detail['ground_truth'])
        self.assertEqual(detail['images'], [])
        self.assertTrue(any('unverified' in warning for warning in detail['warnings']))
        with self.assertRaises(KeyError):
            prediction_image(self.store, run['id'], fixtures.FRAMES[0], 'canonical')

    def test_failed_predictions_never_get_semantic_scores(self):
        run = self.store.import_run({'name': 'Failure fixture', 'model': 'None', 'track': 'historical',
            'records': [{'frame_id': fixtures.FRAMES[0], 'status': 'failed'}]})
        review = build_review(self.store, run['id'])
        self.assertEqual(review['counts']['ineligible'], 1)
        self.assertIsNone(review['frames'][0]['scores'])
        with self.assertRaises(KeyError):
            build_review(self.store, 'not-a-run')
        with self.assertRaises(KeyError):
            build_review(self.store, run['id'], 'not-a-group')


if __name__ == '__main__':
    unittest.main()
