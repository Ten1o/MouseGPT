"""Offline checklist contract tests; verdict fixtures are not scientific labels."""
import copy
import unittest

from mousebench.criteria import validate_criteria, validate_assessments, summarize_assessments
from mousebench.protocol import default_rubric, validate_rubric, rubric_hash, benchmark_index


def spec():
    return [{"id": "visible_relation", "label": "Visible relation", "definition": "Compare the visible views.",
             "allow_not_assessable": True, "group": "grounding"},
            {"id": "task_scope", "label": "Task scope", "definition": "Answer the requested task.",
             "allow_not_assessable": False, "group": "instruction_following"}]


def assessments(first="met", second="met"):
    return {"visible_relation": {"verdict": first, "evidence": "Fixture evidence about visibility."},
            "task_scope": {"verdict": second, "evidence": "Fixture evidence about the task."}}


class CriteriaTests(unittest.TestCase):
    def test_definitions_are_normalized_without_mutating_source(self):
        original = spec()
        original[0]["label"] = "  Visible relation  "
        cleaned = validate_criteria(original)
        self.assertEqual(cleaned[0]["label"], "Visible relation")
        self.assertEqual(original[0]["label"], "  Visible relation  ")
        cleaned[0]["definition"] = "Changed"
        self.assertNotEqual(original[0]["definition"], "Changed")

    def test_duplicate_bad_ids_missing_metadata_and_weights_are_rejected(self):
        for mutation in ({"id": "Task Scope"}, {"allow_not_assessable": "yes"}, {"definition": " "}, {"weight": .5}):
            bad = spec()
            bad[0].update(mutation)
            with self.assertRaises(ValueError):
                validate_criteria(bad)
        with self.assertRaises(ValueError):
            validate_criteria(spec() + [spec()[0]])
        with self.assertRaises(ValueError):
            validate_criteria({})

    def test_assessments_require_every_id_and_no_unknown_criterion(self):
        values = assessments()
        self.assertEqual(validate_assessments(values, spec()), values)
        for bad in ({}, {"visible_relation": values["visible_relation"]}, dict(values, unknown=values["task_scope"])):
            with self.assertRaises(ValueError):
                validate_assessments(bad, spec())

    def test_no_implicit_pass_numeric_scores_or_evidenceless_verdict(self):
        for invalid in ({"verdict": "met", "evidence": " "}, {"verdict": True, "evidence": "Fixture"},
                        {"verdict": 1, "evidence": "Fixture"}, {"verdict": "met", "evidence": "Fixture", "score": 1}):
            values = assessments()
            values["visible_relation"] = invalid
            with self.assertRaises(ValueError):
                validate_assessments(values, spec())
        with self.assertRaises(ValueError):
            validate_assessments(None, spec())

    def test_na_allowed_only_when_definition_permits_it(self):
        values = validate_assessments(assessments("not_assessable"), spec())
        self.assertEqual(values["visible_relation"]["verdict"], "not_assessable")
        with self.assertRaises(ValueError):
            validate_assessments(assessments(second="not_assessable"), spec())

    def test_summary_separates_applicable_na_and_missing(self):
        result = summarize_assessments([assessments(), assessments("unmet", "unmet"), assessments("not_assessable"), None], spec())
        self.assertEqual((result["record_count"], result["assessed_record_count"], result["missing_record_count"]), (4, 3, 1))
        self.assertEqual(result["assessment_coverage"], .75)
        relation = result["criteria"]["visible_relation"]
        self.assertEqual((relation["met"], relation["unmet"], relation["not_assessable"], relation["missing"]), (1, 1, 1, 1))
        self.assertEqual((relation["applicable_count"], relation["met_rate"]), (2, .5))
        self.assertAlmostEqual(result["criteria"]["task_scope"]["met_rate"], 2 / 3)
        self.assertNotIn("score", result)

    def test_all_na_and_legacy_rows_have_null_not_perfect_rate(self):
        only_na = summarize_assessments([assessments("not_assessable")], spec())["criteria"]["visible_relation"]
        self.assertIsNone(only_na["met_rate"])
        self.assertEqual(only_na["applicable_count"], 0)
        legacy = summarize_assessments([None, None], spec())
        self.assertEqual(legacy["assessment_coverage"], 0)
        self.assertIsNone(legacy["criteria"]["task_scope"]["met_rate"])
        empty = summarize_assessments([], spec())
        self.assertIsNone(empty["assessment_coverage"])
        self.assertEqual(validate_assessments({}, []), {})

    def test_rubric_integrates_checklist_without_changing_index(self):
        rubric = default_rubric()
        self.assertEqual(rubric["id"], "mousebench-description-v3-graded-draft")
        self.assertEqual(rubric["scoring"]["version"], "mousebench.graded-factual-f1.v2")
        self.assertEqual(len(rubric["analytic_criteria"]), 8)
        original_hash = rubric_hash(rubric)
        score = {"factual_f1": .5, "structure_score": 1, "instruction_score": .5}
        self.assertAlmostEqual(benchmark_index(score, rubric)["value"], 47.5)
        legacy = copy.deepcopy(rubric)
        del legacy["analytic_criteria"]
        self.assertEqual(validate_rubric(legacy), legacy)
        self.assertAlmostEqual(benchmark_index(score, legacy)["value"], 47.5)
        self.assertNotEqual(rubric_hash(legacy), original_hash)
        rubric["analytic_criteria"][0]["allow_not_assessable"] = "true"
        with self.assertRaises(ValueError):
            validate_rubric(rubric)


if __name__ == "__main__":
    unittest.main()
