"""Graded factual scoring invariants; no providers, database or network."""

import copy
import math
import unittest

from mousebench.evaluation import (
    GRADED_JUDGMENT_SCHEMA_VERSION, GRADED_SCORING_VERSION,
    GRADED_DIAGNOSTICS_VERSION, GRADED_FIELD_WEIGHTS, SCORING_VERSION,
    aggregate_scorecards, score_judgment, prepare_judgment_scorer,
)
from mousebench.protocol import default_rubric, rubric_hash


FRAME = "240420_C10M1_00003972"


def rubric_fixture():
    rubric = copy.deepcopy(default_rubric())
    rubric["id"] = "graded-scoring-test"
    rubric["scoring"].update(version=GRADED_SCORING_VERSION,
                             field_weights=copy.deepcopy(GRADED_FIELD_WEIGHTS), unknown_gt_weight=1)
    return rubric


def reference():
    return {"status": "approved", "facts": [
        {"id": "paw", "text": "A forepaw is lifted close to the floor.", "field": "Limb", "required": True},
        {"id": "tail", "text": "The tail is visible.", "field": "Limb", "required": False},
    ]}


def claim(cid="c1", text="A forepaw is lifted.", field="Limb", correctness=1,
          error_type="none", severity="none", fact_ids=None):
    return {"id": cid, "text": text, "field": field, "correctness": correctness,
            "error_type": error_type, "severity": severity,
            "evidence": "Fixture image evidence.", "fact_ids": ["paw"] if fact_ids is None else fact_ids}


def judgment(rubric, coverage=.75, claims=None, frame=FRAME, run="run-a"):
    return {"schema_version": GRADED_JUDGMENT_SCHEMA_VERSION, "frame_id": frame, "run_id": run,
            "protocol_version": rubric["id"], "track": "controlled",
            "judge": {"provider": "fixture", "model": "test-judge", "version": "v2"},
            "rubric_sha256": rubric_hash(rubric), "input_sha256": "b" * 64,
            "ground_truth_sha256": "c" * 64, "prediction_sha256": "d" * 64,
            "claims": [claim()] if claims is None else claims,
            "fact_coverage": [{"fact_id": "paw", "coverage": coverage,
                               "claim_ids": ["c1"], "evidence": "The stated lift omits its low height."}],
            "structure_score": 1, "instruction_score": .75}


class GradedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rubric = rubric_fixture()
        self.gt = reference()
        self.j = judgment(self.rubric)

    def score(self, j=None, gt=None, rubric=None):
        return score_judgment(j or self.j, gt or self.gt, self.rubric if rubric is None else rubric)

    def test_prepared_scorer_preserves_values_errors_and_private_rubric_copy(self):
        prepared = prepare_judgment_scorer(self.rubric)
        expected = self.score()
        self.assertEqual(prepared(self.j, self.gt), expected)
        for key, change in [('rubric_sha256', 'a' * 64), ('protocol_version', 'wrong'),
                            ('structure_score', .6)]:
            invalid = dict(self.j, **{key: change})
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepared(invalid, self.gt)
        self.rubric['scoring']['field_weights']['Overall'] = 3
        self.assertEqual(prepared(self.j, self.gt), expected)
        with self.assertRaises(ValueError):
            self.score()
        with self.assertRaises(ValueError):
            prepare_judgment_scorer(self.rubric)

    def test_correctness_and_coverage_are_asymmetric(self):
        score = self.score()
        self.assertEqual(score["factual_precision"], 1)
        self.assertEqual(score["required_fact_recall"], .75)
        self.assertAlmostEqual(score["factual_f1"], 6 / 7)
        self.assertEqual(score["supported_fact_ids"], [])
        self.assertEqual(score["partially_covered_fact_ids"], ["paw"])
        self.assertEqual(score["missing_required_fact_ids"], [])
        self.assertEqual(score["denominators"]["weighted_coverage_credit"], .75)

    def test_fully_covered_ids_are_distinct_from_partial_and_missing(self):
        self.j["fact_coverage"][0]["coverage"] = 1
        score = self.score()
        self.assertEqual(score["supported_fact_ids"], ["paw"])
        self.assertEqual(score["factual_f1"], 1)
        self.j["fact_coverage"][0].update(coverage=0, claim_ids=[])
        score = self.score()
        self.assertEqual(score["missing_required_fact_ids"], ["paw"])
        self.assertEqual(score["factual_precision"], 1)
        self.assertEqual(score["factual_f1"], 0)

    def test_unmatched_extra_claim_affects_precision_without_extra_deduction(self):
        self.j["fact_coverage"][0]["coverage"] = 1
        self.j["claims"].append(claim("extra", "The whiskers are moving.", "Others", 0,
                                       "unverifiable", "major", []))
        score = self.score()
        self.assertAlmostEqual(score["factual_precision"], 2 / 3)
        self.assertEqual(score["required_fact_recall"], 1)
        self.assertAlmostEqual(score["factual_f1"], .8)
        self.assertEqual(score["unverifiable_rate"], .5)
        self.assertEqual(score["major_error_rate"], .5)
        self.assertTrue(score["any_major_error"])
        self.j["claims"][1]["severity"] = "minor"
        self.assertEqual(self.score()["factual_f1"], score["factual_f1"])

    def test_field_weights_apply_on_each_side(self):
        self.gt["facts"].append({"id": "behavior", "text": "The mouse is walking.", "field": "Overall", "required": True})
        self.j["claims"].append(claim("behavior", "The mouse is grooming.", "Overall", 0,
                                      "contradicted", "major", ["behavior"]))
        self.j["fact_coverage"][0]["coverage"] = 1
        self.j["fact_coverage"].append({"fact_id": "behavior", "coverage": 0,
                                         "claim_ids": ["behavior"], "evidence": "The behavior is incorrect."})
        score = self.score()
        self.assertAlmostEqual(score["factual_precision"], 1 / 3)
        self.assertAlmostEqual(score["required_fact_recall"], 1 / 3)
        self.assertAlmostEqual(score["factual_f1"], 1 / 3)
        self.assertEqual(score["denominators"]["claim_weight"], 3)
        self.assertEqual(score["denominators"]["required_fact_weight"], 3)

    def test_candidate_field_does_not_change_reference_weight(self):
        self.j["claims"][0]["field"] = "Overall"
        score = self.score()
        self.assertEqual(score["denominators"]["claim_weight"], 2)
        self.assertEqual(score["denominators"]["required_fact_weight"], 1)
        self.assertEqual(score["required_fact_recall"], .75)

    def test_optional_correct_fact_may_earn_precision_without_recall_target(self):
        self.j["claims"].append(claim("tail", "The tail is visible.", fact_ids=["tail"]))
        score = self.score()
        self.assertEqual(score["factual_precision"], 1)
        self.assertEqual(score["required_fact_recall"], .75)
        self.assertEqual(score["denominators"]["required_facts"], 1)

    def test_unassigned_gt_field_has_reported_weight_one(self):
        self.gt["facts"][0].pop("field")
        score = self.score()
        self.assertEqual(score["graded_details"]["unassigned_required_fact_ids"], ["paw"])
        self.assertEqual(score["denominators"]["required_fact_weight"], 1)
        self.assertEqual(score["grounding_diagnostics"]["unassigned_covered_required_facts"], .75)

    def test_legacy_keyword_gt_field_retains_weight_but_candidates_cannot_use_it(self):
        self.gt["facts"][0]["field"] = "Keywords"
        score = self.score()
        self.assertEqual(score["grounding_diagnostics"]["required_fact_recall_by_section"]["Keywords"]["recall"], .75)
        self.j["claims"][0]["field"] = "Keywords"
        with self.assertRaisesRegex(ValueError, "semantic home"):
            self.score()

    def test_normalized_duplicates_count_once_and_links_canonicalize(self):
        self.j["claims"].append(claim("repeat", "  A FOREPAW  is lifted. "))
        self.j["fact_coverage"][0]["claim_ids"] = ["c1", "repeat"]
        original = copy.deepcopy(self.j)
        score = self.score()
        self.assertEqual(score["denominators"]["claims_submitted"], 2)
        self.assertEqual(score["denominators"]["distinct_asserted_claims"], 1)
        self.assertEqual(score["graded_details"]["claims"][0]["aliases"], ["repeat"])
        self.assertEqual(score["graded_details"]["fact_coverage"][0]["claim_ids"], ["c1"])
        self.assertEqual(self.j, original)

    def test_duplicate_assertions_can_merge_consistent_fact_mappings(self):
        self.j["claims"].append(claim("repeat", fact_ids=["tail"]))
        score = self.score()
        self.assertEqual(score["graded_details"]["claims"][0]["fact_ids"], ["paw", "tail"])

    def test_duplicate_claim_ids_fail_even_for_identical_content(self):
        self.j["claims"].append(copy.deepcopy(self.j["claims"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate claim id"):
            self.score()

    def test_duplicate_claims_with_conflicting_fields_or_grades_fail(self):
        for edits in [{"field": "Head"}, {"correctness": .5, "error_type": "contradicted", "severity": "minor"}]:
            with self.subTest(edits=edits):
                j = copy.deepcopy(self.j)
                duplicate = dict(j["claims"][0], id="repeat", **edits)
                j["claims"].append(duplicate)
                with self.assertRaisesRegex(ValueError, "inconsistent fields or grades"):
                    self.score(j)

    def test_coverage_requires_every_required_fact_once(self):
        for rows in [[], self.j["fact_coverage"] * 2, [dict(self.j["fact_coverage"][0], fact_id="tail")]]:
            with self.subTest(rows=rows):
                j = copy.deepcopy(self.j)
                j["fact_coverage"] = rows
                with self.assertRaises(ValueError):
                    self.score(j)

    def test_positive_coverage_requires_nonzero_linked_correctness(self):
        self.j["claims"][0].update(correctness=0, error_type="contradicted", severity="major")
        with self.assertRaisesRegex(ValueError, "nonzero correctness"):
            self.score()
        self.j["fact_coverage"][0]["coverage"] = 0
        self.assertEqual(self.score()["factual_f1"], 0)

    def test_coverage_requires_known_claim_and_matching_fact_mapping(self):
        for claim_ids, fact_ids in [(["absent"], ["paw"]), (["c1"], []), (["c1", "c1"], ["paw"])]:
            with self.subTest(claim_ids=claim_ids, fact_ids=fact_ids):
                j = copy.deepcopy(self.j)
                j["fact_coverage"][0]["claim_ids"] = claim_ids
                j["claims"][0]["fact_ids"] = fact_ids
                with self.assertRaises(ValueError):
                    self.score(j)

    def test_all_four_grade_fields_require_finite_quarters_and_reject_boolean(self):
        for value in [True, False, float("nan"), float("inf"), -.25, 1.25, .3, "0.75"]:
            for field in ["correctness", "coverage", "structure_score", "instruction_score"]:
                with self.subTest(value=value, field=field):
                    j = copy.deepcopy(self.j)
                    if field == "correctness":
                        j["claims"][0][field] = value
                    elif field == "coverage":
                        j["fact_coverage"][0][field] = value
                    else:
                        j[field] = value
                    with self.assertRaises(ValueError):
                        self.score(j)

    def test_error_type_and_severity_must_agree_with_full_correctness(self):
        for changes in [{"error_type": "contradicted"}, {"severity": "minor"}, {"correctness": .75},
                        {"correctness": .75, "error_type": "unverifiable"},
                        {"correctness": .75, "severity": "major"}]:
            with self.subTest(changes=changes):
                j = copy.deepcopy(self.j)
                j["claims"][0].update(changes)
                with self.assertRaises(ValueError):
                    self.score(j)

    def test_mixed_diagnostics_overlap_without_extra_deduction(self):
        self.j["claims"][0].update(correctness=.5, error_type="mixed", severity="major")
        score = self.score()
        self.assertEqual(score["contradiction_rate"], 1)
        self.assertEqual(score["unverifiable_rate"], 1)
        self.assertEqual(score["major_error_rate"], 1)
        self.assertAlmostEqual(score["factual_f1"], .6)

    def test_legacy_label_and_supported_ids_are_not_valid_graded_inputs(self):
        j = copy.deepcopy(self.j)
        j["claims"][0]["label"] = "supported"
        with self.assertRaises(ValueError):
            self.score(j)
        j = copy.deepcopy(self.j)
        j["supported_fact_ids"] = ["paw"]
        with self.assertRaises(ValueError):
            self.score(j)

    def test_rubric_must_be_present_bound_and_select_graded_policy(self):
        with self.assertRaisesRegex(ValueError, "bound rubric"):
            score_judgment(self.j, self.gt)
        j = copy.deepcopy(self.j)
        j["rubric_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "rubric_sha256"):
            self.score(j)
        changed = copy.deepcopy(self.rubric)
        changed["scoring"]["version"] = SCORING_VERSION
        with self.assertRaisesRegex(ValueError, "scoring.version"):
            self.score(rubric=changed)
        j = copy.deepcopy(self.j)
        j["protocol_version"] = "different-policy"
        with self.assertRaisesRegex(ValueError, "protocol_version"):
            self.score(j)

    def test_weight_policy_cannot_silently_change(self):
        for value in [True, 0, -1, float("nan"), 3]:
            with self.subTest(value=value):
                changed = copy.deepcopy(self.rubric)
                changed["scoring"]["field_weights"]["Overall"] = value
                with self.assertRaises(ValueError):
                    self.score(rubric=changed)
        changed = copy.deepcopy(self.rubric)
        changed["scoring"]["field_weights"].pop("Keywords")
        with self.assertRaises(ValueError):
            self.score(rubric=changed)
        changed = copy.deepcopy(self.rubric)
        changed["scoring"]["unknown_gt_weight"] = 2
        with self.assertRaises(ValueError):
            self.score(rubric=changed)

    def test_empty_claims_score_zero_and_no_required_gt_is_unscorable(self):
        self.j["claims"] = []
        self.j["fact_coverage"][0].update(coverage=0, claim_ids=[])
        score = self.score()
        self.assertEqual((score["factual_precision"], score["required_fact_recall"], score["factual_f1"]), (0, 0, 0))
        self.assertFalse(score["grounding_diagnostics"]["fully_grounded_nonempty"])
        self.gt["facts"][0]["required"] = False
        self.j["fact_coverage"] = []
        score = self.score()
        self.assertFalse(score["evaluable"])
        self.assertIsNone(score["factual_f1"])
        self.assertIsNone(score["grounding_diagnostics"]["complete_required_coverage"])

    def test_section_diagnostics_and_aggregate_preserve_fractional_credit(self):
        a = self.score()
        b_j = copy.deepcopy(self.j)
        b_j["frame_id"] = "240507_C8M2_00053640"
        b_j["fact_coverage"][0]["coverage"] = .25
        b = self.score(b_j)
        result = aggregate_scorecards([a, b])
        card = result["runs"]["run-a"]
        self.assertEqual(result["scoring_version"], GRADED_SCORING_VERSION)
        self.assertEqual(card["grounding_diagnostics"]["version"], GRADED_DIAGNOSTICS_VERSION)
        self.assertEqual(card["grounding_diagnostics"]["sections"]["Limb"]["micro_recall"], .5)
        self.assertEqual(card["dimension_means"]["major_error_rate"], 0)
        self.assertEqual(card["grounding_diagnostics"]["frame_indicators"]["any_major_error"]["rate"], 0)

    def test_paired_graded_comparison_requires_matching_versions(self):
        a = self.score()
        b_j = copy.deepcopy(self.j)
        b_j["run_id"] = "run-b"
        b_j["fact_coverage"][0]["coverage"] = .25
        result = aggregate_scorecards([a, self.score(b_j)])
        self.assertEqual(result["comparisons"][0]["paired_cluster_bootstrap"]["cluster_count"], 1)
        self.assertIsNone(result["comparisons"][0]["paired_cluster_bootstrap"]["interval"])
        legacy = self.legacy_score()
        with self.assertRaisesRegex(ValueError, "different protocol, track or scoring versions"):
            aggregate_scorecards([a, dict(legacy, run_id="legacy")])

    def test_zero_common_frames_retains_graded_diagnostic_version(self):
        a = self.score()
        b_j = copy.deepcopy(self.j)
        b_j.update(run_id="run-b", frame_id="240507_C8M2_00053640")
        result = aggregate_scorecards([a, self.score(b_j)])
        self.assertEqual(result["runs"]["run-a"]["common_frame_grounding_diagnostics"]["version"], GRADED_DIAGNOSTICS_VERSION)

    def legacy_score(self):
        legacy = copy.deepcopy(self.j)
        legacy.update(schema_version="mousebench.judgment.v1",
                      claims=[{"text": "A forepaw is lifted.", "label": "supported", "fact_ids": ["paw"]}],
                      supported_fact_ids=["paw"])
        legacy.pop("fact_coverage")
        return score_judgment(legacy, self.gt)

    def test_binary_scorer_and_aggregation_remain_unchanged(self):
        score = self.legacy_score()
        self.assertEqual(score["scoring_version"], SCORING_VERSION)
        self.assertEqual(score["factual_f1"], 1)
        self.assertNotIn("graded_details", score)
        self.assertNotIn("major_error_rate", score)
        aggregate = aggregate_scorecards([score])
        self.assertNotIn("major_error_rate", aggregate["runs"]["run-a"]["dimension_means"])


if __name__ == "__main__":
    unittest.main()
