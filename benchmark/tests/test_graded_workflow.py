"""Graded judge packet -> ingestion -> scorecard -> blinded calibration, offline."""
import copy
import json
import unittest
import test_workflow as fixtures
from mousebench.protocol import default_rubric
from mousebench.workflow import result_schema


class GradedWorkflowTests(unittest.TestCase):
    setUp = fixtures.WorkflowTests.setUp
    tearDown = fixtures.WorkflowTests.tearDown
    inputs = fixtures.WorkflowTests.inputs
    batch = fixtures.WorkflowTests.batch
    receipt = fixtures.WorkflowTests.receipt
    approve = fixtures.WorkflowTests.approve
    candidate = fixtures.WorkflowTests.candidate

    def prepare(self):
        self.inputs()
        self.content['facts'][0]['text'] = 'The head faces left and is lowered.'
        self.approve()
        saved = self.workflow.save_rubric({'content': default_rubric(), 'author': 'Fixture'})
        rubric = self.workflow.freeze_rubric({'id': saved['id'], 'reviewer': 'Fixture'})
        run = self.candidate()
        batch = self.batch('judge', run_id=run, rubric_id=rubric['id'])
        return rubric, run, batch['jobs'][0]['id']

    def judgment(self):
        return {'claims': [{'id': 'c1', 'text': 'The head faces left.', 'field': 'Head',
            'correctness': 1, 'evidence': 'Synthetic fixture direction.', 'error_type': 'none',
            'severity': 'none', 'fact_ids': ['f1']}],
            'fact_coverage': [{'fact_id': 'f1', 'coverage': .75, 'claim_ids': ['c1'],
                'evidence': 'Direction is present, lowered position omitted in this fixture.'}],
            'structure_score': 1, 'instruction_score': .5, 'issues': [],
            'criterion_assessments': {c['id']: {'verdict': 'met', 'evidence': 'Synthetic fixture only.'}
                for c in default_rubric()['analytic_criteria']}}

    def test_packet_and_roundtrip_use_graded_semantics(self):
        rubric, run, job = self.prepare()
        packet = self.workflow.job(job)['packet']
        schema = packet['expected_result_schema']
        self.assertIn('fact_coverage', schema['required'])
        self.assertNotIn('supported_fact_ids', schema['properties'])
        self.assertEqual(schema['properties']['claims']['items']['properties']['correctness']['enum'], [0,.25,.5,.75,1])
        before = self.judgment()
        result = self.workflow.complete(self.receipt(job, before))
        self.assertEqual(result['status'], 'succeeded')
        self.assertTrue(self.workflow.complete(self.receipt(job, before))['duplicate'])
        stored = self.store.export('judgments')[0]
        self.assertEqual(stored['schema_version'], 'mousebench.judgment.v2')
        self.assertEqual(stored['scores']['factual_precision'], 1)
        self.assertEqual(stored['scores']['required_fact_recall'], .75)
        self.assertAlmostEqual(stored['scores']['factual_f1'], 6/7)
        cards = self.workflow.scorecards()
        card = cards['models'][0]
        self.assertEqual(card['scoring_version'], 'mousebench.graded-factual-f1.v2')
        self.assertAlmostEqual(card['dimension_means']['factual_f1'], 6/7)
        self.assertAlmostEqual(card['benchmark_index'], 100*(6/7)*.95)
        self.assertEqual(card['dimension_means']['major_error_rate'], 0)
        self.assertEqual(before, self.judgment())

    def test_blinded_grades_join_authoritative_verdicts(self):
        rubric, run, job = self.prepare()
        self.workflow.complete(self.receipt(job, self.judgment()))
        packet = self.workflow.calibration_template(run)
        self.assertEqual(len(packet['records']), 4)
        dimensions = {r['dimension'] for r in packet['records']}
        self.assertEqual(dimensions, {'factual_correctness','required_fact_coverage','structure_score','instruction_score'})
        self.assertTrue(all('judge_label' not in r and 'judge' not in r for r in packet['records']))
        records = copy.deepcopy(packet['records'])
        for row in records:
            row.update(split='heldout', human_verified=True, verified_by='Fixture expert', human_label=.75)
        result = self.workflow.import_calibration({'rubric_id':rubric['id'], 'records':records})
        metrics = result['report']['splits']['heldout']['metrics']
        self.assertEqual(metrics['factual_correctness']['mean_absolute_error'], .25)
        self.assertEqual(metrics['required_fact_coverage']['mean_absolute_error'], 0)
        self.assertFalse(result['report']['reliability_certified'])
        bad = copy.deepcopy(records)
        bad[0]['judge_label'] = 0
        with self.assertRaisesRegex(ValueError, 'cannot override'):
            self.workflow.import_calibration({'rubric_id':rubric['id'], 'records':bad})

    def test_invalid_or_legacy_result_is_atomic(self):
        _, _, job = self.prepare()
        wrong = self.judgment()
        wrong['claims'][0]['correctness'] = .7
        with self.assertRaises(ValueError):
            self.workflow.complete(self.receipt(job, wrong))
        self.assertEqual(self.workflow.job(job)['status'], 'pending')
        self.assertEqual(self.store.export('judgments'), [])
        wrong = self.judgment()
        wrong['supported_fact_ids'] = ['f1']
        with self.assertRaisesRegex(ValueError, 'legacy'):
            self.workflow.complete(self.receipt(job, wrong))
        self.assertEqual(self.store.export('judgments'), [])

    def test_direct_import_cannot_hide_missing_checklist(self):
        _, _, job = self.prepare()
        self.workflow.complete(self.receipt(job, self.judgment()))
        stored = self.store.export('judgments')[0]
        raw = {key:stored[key] for key in ('schema_version','frame_id','run_id','protocol_version',
            'judge','rubric_sha256','ground_truth_sha256','prediction_sha256','input_sha256',
            'claims','fact_coverage','structure_score','instruction_score')}
        with self.assertRaisesRegex(ValueError, 'criterion'):
            self.store.import_judgments({'records':[raw]})

    def test_direct_import_cannot_bypass_rubric_freeze(self):
        _, _, job = self.prepare()
        self.workflow.complete(self.receipt(job, self.judgment()))
        stored = self.store.export('judgments')[0]
        content = default_rubric()
        content['id'] += '-unfrozen'
        draft = self.workflow.save_rubric({'content': content, 'author': 'Fixture'})
        raw = {key:stored[key] for key in ('schema_version','frame_id','run_id','protocol_version',
            'judge','rubric_sha256','ground_truth_sha256','prediction_sha256','input_sha256',
            'claims','fact_coverage','structure_score','instruction_score','criterion_assessments')}
        raw.update(protocol_version=content['id'], rubric_sha256=draft['sha256'])
        with self.assertRaisesRegex(ValueError, 'Freeze'):
            self.store.import_judgments({'records':[raw]})
        self.assertEqual(len(self.store.export('judgments')), 1)


if __name__ == '__main__':
    unittest.main()
