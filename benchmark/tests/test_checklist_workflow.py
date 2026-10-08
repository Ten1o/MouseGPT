"""New rubric jobs require evidence while legacy imports retain missing coverage."""
import copy
import unittest
import test_workflow as fixtures


class ChecklistWorkflowTests(unittest.TestCase):
    setUp = fixtures.WorkflowTests.setUp
    tearDown = fixtures.WorkflowTests.tearDown
    inputs = fixtures.WorkflowTests.inputs
    frozen_rubric = fixtures.WorkflowTests.frozen_rubric
    batch = fixtures.WorkflowTests.batch
    receipt = fixtures.WorkflowTests.receipt
    approve = fixtures.WorkflowTests.approve
    candidate = fixtures.WorkflowTests.candidate
    judgment = fixtures.WorkflowTests.judgment

    def prepare(self):
        self.inputs()
        self.approve()
        rubric = self.frozen_rubric()
        run = self.candidate()
        batch = self.batch('judge', run_id=run, rubric_id=rubric['id'])
        return batch['jobs'][0]['id']

    def test_new_job_requires_all_evidence_backed_criteria(self):
        job_id = self.prepare()
        packet = self.workflow.job(job_id)['packet']
        self.assertIn('criterion_assessments', packet['expected_result_schema']['required'])
        result = self.judgment()
        del result['criterion_assessments']
        with self.assertRaises(ValueError):
            self.workflow.complete(self.receipt(job_id, result))
        self.assertEqual(self.workflow.job(job_id)['status'], 'pending')
        self.assertEqual(self.workflow.scorecards()['models'], [])

    def test_nonapplicable_verdict_cannot_hide_required_criterion(self):
        job_id = self.prepare()
        result = self.judgment()
        result['criterion_assessments']['terminology']['verdict'] = 'not_assessable'
        with self.assertRaises(ValueError):
            self.workflow.complete(self.receipt(job_id, result))

    def test_saved_results_expose_checklist_and_preserve_receipt(self):
        job_id = self.prepare()
        result = self.judgment()
        before = copy.deepcopy(result)
        self.workflow.complete(self.receipt(job_id, result))
        self.assertEqual(result, before)
        self.assertTrue(self.workflow.complete(self.receipt(job_id, result))['duplicate'])
        card = self.workflow.scorecards()['models'][0]
        summary = card['criterion_results']
        self.assertEqual(summary['assessed_record_count'], 1)
        self.assertEqual(summary['missing_record_count'], 0)
        self.assertEqual(summary['criteria']['terminology']['met_rate'], 1)
        self.assertEqual(card['grounding_diagnostics']['diagnosed_frame_count'], 1)


if __name__ == '__main__':
    unittest.main()
