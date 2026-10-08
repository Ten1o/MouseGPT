"""Offline scientific invariants for editable rubrics and verified calibration."""
import copy
import unittest

from mousebench.protocol import (
    benchmark_index, calibration_report, default_rubric, rubric_hash,
    validate_calibration_record, validate_rubric,
)

JUDGE = {"provider": "fixture", "model": "fixture-judge", "version": "1"}
FRAME = "240420_C10M1_00003972"


def calibration(kind="claim", **changes):
    record = {"id": "c1", "kind": kind, "frame_id": FRAME,
              "recording_id": "240420_C10M1", "split": "heldout",
              "input_sha256": "a" * 64, "prediction_sha256": "b" * 64,
              "rubric_sha256": rubric_hash(default_rubric()), "judge": JUDGE,
              "human_verified": True, "verified_by": "fixture expert",
              "human_label": "supported", "judge_label": "supported",
              "claim_id": "head", "claim_text": "The head is lowered."}
    if kind == "pairwise":
        record.update(other_prediction_sha256="c" * 64, human_label="a", judge_label="a")
    elif kind == "dimension":
        record.update(dimension="structure_score", human_label=1, judge_label=.75)
    record.update(changes)
    return record


class RubricTests(unittest.TestCase):
    def test_hash_is_semantic_order_independent_and_detects_weights(self):
        rubric = default_rubric()
        reordered = dict(reversed(list(rubric.items())))
        self.assertEqual(rubric_hash(rubric), rubric_hash(reordered))
        changed = copy.deepcopy(rubric)
        changed["scoring"]["index"].update(base=.7, structure_weight=.2)
        self.assertNotEqual(rubric_hash(rubric), rubric_hash(changed))

    def test_normalization_does_not_mutate_and_rejects_unknown_arithmetic(self):
        rubric = default_rubric()
        original = copy.deepcopy(rubric)
        self.assertEqual(validate_rubric(rubric), original)
        self.assertEqual(rubric, original)
        for patch in ({"base": .9}, {"base": -.1}, {"base": True}, {"base": float("nan")}, {"formula": "new unimplemented formula"}):
            broken = copy.deepcopy(rubric)
            broken["scoring"]["index"].update(patch)
            with self.assertRaises(ValueError):
                validate_rubric(broken)
        rubric["fields"].remove("Head")
        with self.assertRaises(ValueError):
            validate_rubric(rubric)

    def test_zero_facts_cannot_be_masked_by_perfect_style(self):
        scores = {"factual_f1": 0, "structure_score": 1, "instruction_score": 1}
        self.assertEqual(benchmark_index(scores)["value"], 0)
        scores["factual_f1"] = 1
        self.assertEqual(benchmark_index(scores)["value"], 100)
        scores.update(structure_score=0, instruction_score=0)
        self.assertEqual(benchmark_index(scores)["value"], 80)
        self.assertTrue(benchmark_index(scores)["provisional"])

    def test_missing_dimension_and_unscorable_gt_are_not_imputed(self):
        score = benchmark_index({"factual_f1": .8, "structure_score": 1})
        self.assertIsNone(score["value"])
        self.assertEqual(score["reason"], "missing_required_dimension")
        score = benchmark_index({"factual_f1": .8, "structure_score": 1, "instruction_score": 1, "evaluable": False})
        self.assertIsNone(score["value"])
        with self.assertRaises(ValueError):
            benchmark_index({"factual_f1": 1.01, "structure_score": 1, "instruction_score": 1})


class CalibrationTests(unittest.TestCase):
    def test_requires_human_verification_and_exact_binding(self):
        for change in ({"human_verified": False}, {"verified_by": ""}, {"prediction_sha256": "filename.jpg"},
                       {"rubric_sha256": "f" * 64}, {"recording_id": "another-recording"},
                       {"human_label": "correct"}, {"split": "train"}):
            with self.assertRaises(ValueError):
                validate_calibration_record(calibration(**change), default_rubric())

    def test_empty_and_development_only_report_no_validation(self):
        empty = calibration_report([])
        self.assertEqual(empty["n"], 0)
        self.assertFalse(empty["reliability_certified"])
        self.assertIsNone(empty["splits"]["heldout"]["metrics"]["claim"]["exact_agreement"])
        report = calibration_report([calibration(split="development")])
        self.assertEqual(report["status"], "needs_heldout_evidence")

    def test_confusion_kappa_and_clustered_uncertainty(self):
        first = calibration()
        second = calibration(id="c2", frame_id="240507_C8M2_00053640", recording_id="240507_C8M2",
                             human_label="contradicted", judge_label="supported")
        report = calibration_report([first, second])
        metric = report["splits"]["heldout"]["metrics"]["claim"]
        self.assertEqual(metric["exact_agreement"], .5)
        self.assertEqual(metric["cohens_kappa"], 0)
        self.assertEqual(metric["confusion_matrix"]["contradicted"]["supported"], 1)
        self.assertEqual(metric["agreement_cluster_bootstrap"]["cluster_count"], 2)
        self.assertEqual(metric["agreement_cluster_bootstrap"]["replicates"], 1000)
        self.assertEqual(report["disagreement_ids"], ["c2"])
        self.assertFalse(report["reliability_certified"])

    def test_single_label_kappa_is_undefined_not_perfect(self):
        report = calibration_report([calibration()])
        metric = report["splits"]["heldout"]["metrics"]["claim"]
        self.assertEqual(metric["exact_agreement"], 1)
        self.assertIsNone(metric["cohens_kappa"])
        self.assertIsNone(metric["agreement_cluster_bootstrap"]["interval"])

    def test_renamed_duplicate_and_recording_leakage_are_rejected(self):
        first = calibration()
        for second in (calibration(id="new-id"), calibration(id="new-id", claim_id="new-claim"),
                       calibration(id="new-id", claim_id="new", claim_text="The paw is hidden.", split="development")):
            with self.assertRaises(ValueError):
                calibration_report([first, second])

    def test_pairwise_hash_orientation_and_swapped_duplicates(self):
        first = calibration("pairwise")
        report = calibration_report([first])
        self.assertEqual(report["splits"]["heldout"]["metrics"]["pairwise"]["exact_agreement"], 1)
        swapped = calibration("pairwise", id="swapped", prediction_sha256="c" * 64,
                              other_prediction_sha256="b" * 64, human_label="b", judge_label="b")
        with self.assertRaises(ValueError):
            calibration_report([first, swapped])

    def test_paper_correctness_labels_are_separate_from_claim_labels(self):
        row = calibration("correctness", human_label="correct", judge_label="incorrect")
        report = calibration_report([row])
        metric = report["splits"]["heldout"]["metrics"]["correctness"]
        self.assertEqual(metric["exact_agreement"], 0)
        self.assertEqual(metric["confusion_matrix"]["correct"]["incorrect"], 1)

    def test_rating_error_and_selected_judge(self):
        report = calibration_report([calibration("dimension")], judge=JUDGE)
        metric = report["splits"]["heldout"]["metrics"]["structure_score"]
        self.assertEqual(metric["mean_absolute_error"], .25)
        self.assertEqual(metric["signed_bias"], -.25)
        with self.assertRaises(ValueError):
            calibration_report([calibration()], judge=dict(JUDGE, version="2"))
        with self.assertRaises(ValueError):
            calibration_report([calibration(), calibration(id="c2", claim_id="tail", claim_text="Tail extended.", judge=dict(JUDGE, version="2"))])


if __name__ == "__main__":
    unittest.main()
