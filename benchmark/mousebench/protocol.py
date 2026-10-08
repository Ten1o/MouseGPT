"""Versionable MouseGPT rubric, transparent index and human/judge calibration.

All functions are offline. They calculate agreement, not scientific validity.
A calibration record asserts a human verified an exact output binding; callers
must verify that assertion and its stored hashes before accepting the record.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

from .evaluation import OUTPUT_FIELDS, CLAIM_LABELS

RUBRIC_SCHEMA = "mousebench.rubric.v1"
CALIBRATION_SCHEMA = "mousebench.calibration.v1"
INDEX_VERSION = "mousebench.grounded-quality.v1"
LEGACY_SCORING_VERSION = "mousebench.factual-f1.v1"
GRADED_SCORING_VERSION = "mousebench.graded-factual-f1.v2"
GRADED_INDEX_VERSION = "mousebench.graded-quality.v2"
GRADED_FIELD_WEIGHTS = {"Overall": 2, "Head": 1, "Limb": 1, "Torso": 1, "Others": .5, "Keywords": 1}
QUARTERS = (0, .25, .5, .75, 1)
_HEX = re.compile(r"^[a-fA-F0-9]{64}$")
from .identity import FRAME_ID
_FRAME = FRAME_ID
_CATEGORICAL = {"claim": sorted(CLAIM_LABELS), "pairwise": ["a", "b", "tie"],
                "correctness": ["correct", "incorrect"]}
_DIMENSIONS = ("structure_score", "instruction_score")
_CALIBRATION_DIMENSIONS = _DIMENSIONS + ("factual_correctness", "required_fact_coverage")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _quarter(value, name):
    value = _unit(value, name)
    if value not in QUARTERS:
        raise ValueError(name + " must be one of 0, 0.25, 0.5, 0.75 or 1")
    return value


def _graded(rubric):
    return isinstance(rubric, dict) and isinstance(rubric.get("scoring"), dict) and rubric["scoring"].get("version") == GRADED_SCORING_VERSION


def _hash(value, name):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return value.lower()


def _judge(value):
    if not isinstance(value, dict) or set(value) != {"provider", "model", "version"}:
        raise ValueError("judge must contain exactly provider, model and version")
    return {key: _text(value[key], "judge." + key) for key in ("provider", "model", "version")}


def validate_rubric(content):
    """Validate supported scoring semantics without silently changing weights.

    Extra explanatory metadata is preserved and hashed. The caller freezes an
    immutable copy when publishing a version. Unsupported arithmetic is rejected
    until implemented in code, even if a draft's prose describes it.
    """
    if not isinstance(content, dict):
        raise ValueError("rubric must be an object")
    result = copy.deepcopy(content)
    result.setdefault("schema_version", RUBRIC_SCHEMA)
    if result["schema_version"] != RUBRIC_SCHEMA:
        raise ValueError("unsupported rubric schema_version")
    for key in ("id", "task"):
        result[key] = _text(result.get(key), key)
    if result.get("status") not in {"draft_requires_calibration", "draft", "frozen", "released"}:
        raise ValueError("unsupported rubric status")
    if result.get("fields") != list(OUTPUT_FIELDS):
        raise ValueError("rubric must retain the ordered six MouseGPT fields")
    graded = _graded(result)
    verdict_field = "error_types" if graded else "claim_verdicts"
    expected_verdicts = {"none", "contradicted", "unverifiable", "mixed"} if graded else CLAIM_LABELS
    verdicts = result.get(verdict_field)
    if not isinstance(verdicts, dict) or set(verdicts) != expected_verdicts:
        raise ValueError(verdict_field + " must define " + ", ".join(sorted(expected_verdicts)))
    for key, value in verdicts.items():
        _text(value, verdict_field + "." + key)
    _text(result.get("required_fact_coverage"), "required_fact_coverage")
    anchor_fields = ["structure_score_anchors", "instruction_score_anchors"]
    if graded:
        anchor_fields += ["correctness_score_anchors", "coverage_score_anchors"]
    for field in anchor_fields:
        anchors = result.get(field)
        if not isinstance(anchors, dict) or not {"0", "0.5", "1"} <= set(anchors):
            raise ValueError(field + " must define at least 0, 0.5 and 1 anchors")
        if graded and set(anchors) != {"0", "0.25", "0.5", "0.75", "1"}:
            raise ValueError(field + " must define exactly the five quarter-step anchors")
        for key, value in anchors.items():
            try:
                _unit(float(key), field + " anchor")
            except (ValueError, TypeError):
                raise ValueError(field + " anchors must be numbers in [0, 1]")
            _text(value, field + "." + key)
    rules = result.get("rules")
    if not isinstance(rules, list) or not rules or any(not isinstance(x, str) or not x.strip() for x in rules):
        raise ValueError("rules must be a nonempty list of nonempty strings")
    scoring = result.get("scoring")
    scoring_version = scoring.get("version", LEGACY_SCORING_VERSION) if isinstance(scoring, dict) else None
    if scoring_version not in {LEGACY_SCORING_VERSION, GRADED_SCORING_VERSION}:
        raise ValueError("unsupported scoring version")
    expected_primary = "graded factual F1" if graded else "grounded factual F1"
    if scoring.get("primary") != expected_primary or scoring.get("components_separate") is not True:
        raise ValueError("scoring must retain " + expected_primary + " and separate components")
    if graded:
        weights = scoring.get("field_weights")
        if not isinstance(weights, dict) or set(weights) != set(GRADED_FIELD_WEIGHTS):
            raise ValueError("graded scoring requires the six specified field weights")
        for name, expected in GRADED_FIELD_WEIGHTS.items():
            value = weights[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                raise ValueError("unsupported graded field weight for " + name)
        if isinstance(scoring.get("unknown_gt_weight"), bool) or scoring.get("unknown_gt_weight") != 1:
            raise ValueError("graded scoring requires unknown_gt_weight 1")
    index = scoring.get("index")
    expected_index = GRADED_INDEX_VERSION if graded else INDEX_VERSION
    if not isinstance(index, dict) or index.get("version") != expected_index:
        raise ValueError("scoring.index must match the selected scoring version")
    if index.get("formula") != "100 * factual_f1 * (base + structure_weight * structure_score + instruction_weight * instruction_score)":
        raise ValueError("unsupported index formula")
    weights = [_unit(index.get(key), "scoring.index." + key) for key in ("base", "structure_weight", "instruction_weight")]
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("index multipliers must sum to 1")
    if weights[0] < .5:
        raise ValueError("the grounded-quality index requires a factual base of at least 0.5")
    for key, value in zip(("base", "structure_weight", "instruction_weight"), weights):
        index[key] = value
    if "analytic_criteria" in result:
        from .criteria import validate_criteria
        result["analytic_criteria"] = validate_criteria(result["analytic_criteria"])
    calibration = result.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be an object")
    # Reject non-JSON metadata, NaN and infinities anywhere, not just weights.
    json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return result


def default_rubric():
    return validate_rubric(json.loads((Path(__file__).resolve().parents[1] / "protocol" / "rubric.json").read_text()))


def rubric_hash(content):
    normalized = validate_rubric(content)
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def benchmark_index(scores, rubric=None):
    """Return one frame's proposed 0..100 secondary quality index.

    Compute this for each eligible frame, then average those values. Applying
    the nonlinear formula to averaged components is not the same calculation.
    Missing judgments are not zero; execution completion is reported separately.
    """
    return prepare_benchmark_index(rubric)(scores)


def prepare_benchmark_index(rubric=None):
    """Capture a validated immutable-by-ownership policy for one aggregation pass."""
    normalized = validate_rubric(rubric) if rubric is not None else default_rubric()
    sha256 = rubric_hash(normalized)
    return lambda scores: _benchmark_index(scores, normalized, sha256)


def _benchmark_index(scores, rubric, sha256):
    if not isinstance(scores, dict):
        raise ValueError("scores must be an object")
    spec = rubric["scoring"]["index"]
    selected_scoring = rubric["scoring"].get("version", LEGACY_SCORING_VERSION)
    if scores.get("scoring_version") not in (None, selected_scoring):
        raise ValueError("score and rubric scoring versions differ; historical scores cannot be relabeled")
    result = {"version": spec["version"], "rubric_sha256": sha256, "value": None,
              "primary": rubric["scoring"]["primary"], "primary_value": scores.get("factual_f1"),
              "components": {key: scores.get(key) for key in ("factual_f1", *_DIMENSIONS)},
              "formula": spec["formula"], "weights": {key: spec[key] for key in ("base", "structure_weight", "instruction_weight")},
              "provisional": True, "status": "unscorable", "reason": None}
    if scores.get("evaluable") is False:
        result["reason"] = scores.get("reason") or "unevaluable_record"
        return result
    if any(result["components"][key] is None for key in result["components"]):
        result["reason"] = "missing_required_dimension"
        return result
    parts = {key: (_quarter(value, key) if _graded(rubric) and key in _DIMENSIONS else _unit(value, key))
             for key, value in result["components"].items()}
    result.update(value=100 * parts["factual_f1"] * (spec["base"] + spec["structure_weight"] * parts["structure_score"] + spec["instruction_weight"] * parts["instruction_score"]),
                  status="scored", components=parts)
    return result


def validate_calibration_record(record, rubric=None, judge=None):
    """Normalize a verified, exact-output human/judge comparison.

    split is development or heldout, with recording-level separation checked by
    calibration_report. 'human_verified' is a provenance declaration, never an
    inference from an old spreadsheet row or an AI-generated label.
    """
    if not isinstance(record, dict):
        raise ValueError("calibration record must be an object")
    item = copy.deepcopy(record)
    item.setdefault("schema_version", CALIBRATION_SCHEMA)
    if item["schema_version"] != CALIBRATION_SCHEMA:
        raise ValueError("unsupported calibration schema_version")
    for key in ("id", "frame_id", "recording_id", "verified_by"):
        item[key] = _text(item.get(key), key)
    if not _FRAME.fullmatch(item["frame_id"]):
        raise ValueError("invalid frame_id")
    if item["recording_id"] != item["frame_id"].rsplit("_", 1)[0]:
        raise ValueError("recording_id must match the frame capture date and group")
    if item.get("human_verified") is not True:
        raise ValueError("calibration requires explicitly human-verified output/label binding")
    if item.get("split") not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    for key in ("input_sha256", "prediction_sha256", "rubric_sha256"):
        item[key] = _hash(item.get(key), key)
    if rubric is not None and item["rubric_sha256"] != rubric_hash(rubric):
        raise ValueError("calibration record is bound to a different rubric")
    item["judge"] = _judge(item.get("judge"))
    if judge is not None and item["judge"] != _judge(judge):
        raise ValueError("calibration judge differs from selected judge")
    kind = item.get("kind")
    if kind in _CATEGORICAL:
        for key in ("human_label", "judge_label"):
            if not isinstance(item.get(key), str) or item[key] not in _CATEGORICAL[kind]:
                raise ValueError(key + " is invalid for " + kind)
        if kind == "claim":
            for key in ("claim_id", "claim_text"):
                item[key] = _text(item.get(key), key)
        elif kind == "pairwise":
            item["other_prediction_sha256"] = _hash(item.get("other_prediction_sha256"), "other_prediction_sha256")
            if item["other_prediction_sha256"] == item["prediction_sha256"]:
                raise ValueError("pairwise comparison requires two distinct outputs")
            # a/b always refer to these hash slots, not provider display order.
            item["label_orientation"] = "a=prediction_sha256;b=other_prediction_sha256"
    elif kind == "dimension":
        dimension = item.get("dimension")
        if dimension not in _CALIBRATION_DIMENSIONS:
            raise ValueError("unsupported calibration dimension")
        if dimension == "factual_correctness":
            for key in ("claim_id", "claim_text"):
                item[key] = _text(item.get(key), key)
        elif dimension == "required_fact_coverage":
            item["fact_id"] = _text(item.get("fact_id"), "fact_id")
            if "fact_text" in item:
                item["fact_text"] = _text(item["fact_text"], "fact_text")
        graded = _graded(rubric) or item.get("scoring_version") == GRADED_SCORING_VERSION
        for key in ("human_label", "judge_label"):
            item[key] = (_quarter if graded else _unit)(item.get(key), key)
    else:
        raise ValueError("kind must be claim, pairwise, correctness or dimension")
    json.dumps(item, allow_nan=False)
    return item


def _quantile(values, q):
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (idx - lo) * (values[hi] - values[lo])


def _cluster_interval(records, values):
    groups = {}
    for record, value in zip(records, values):
        groups.setdefault(record["recording_id"], []).append(value)
    result = {"unit": "recording_id", "cluster_count": len(groups), "confidence_level": .95,
              "replicates": 0, "interval": None, "seed": 395}
    if len(groups) < 2:
        result["reason"] = "at_least_two_recordings_required"
        return result
    sums = [(sum(items), len(items)) for _, items in sorted(groups.items())]
    rng = random.Random(395)
    samples = []
    for _ in range(1000):
        sampled = [rng.choice(sums) for _ in sums]
        samples.append(sum(v for v, n in sampled) / sum(n for v, n in sampled))
    result.update(replicates=1000, interval=[_quantile(samples, .025), _quantile(samples, .975)], reason=None)
    return result


def _agreement(records, labels=None):
    n = len(records)
    base = {"n": n, "frame_count": len({r["frame_id"] for r in records}),
            "output_count": len({r["prediction_sha256"] for r in records} | {r["other_prediction_sha256"] for r in records if "other_prediction_sha256" in r}),
            "recording_count": len({r["recording_id"] for r in records})}
    if labels is None:
        errors = [abs(r["human_label"] - r["judge_label"]) for r in records]
        base.update(mean_absolute_error=statistics.mean(errors) if n else None,
                    signed_bias=statistics.mean([r["judge_label"] - r["human_label"] for r in records]) if n else None,
                    within_quarter_agreement=sum(value <= .25 for value in errors) / n if n else None,
                    error_cluster_bootstrap=_cluster_interval(records, errors))
        return base
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for r in records:
        confusion[r["human_label"]][r["judge_label"]] += 1
    correct = [int(r["human_label"] == r["judge_label"]) for r in records]
    observed = sum(correct) / n if n else None
    chance = sum(sum(confusion[a].values()) * sum(confusion[b][a] for b in labels) for a in labels) / (n * n) if n else None
    human_counts = {label: sum(confusion[label].values()) for label in labels}
    base.update(exact_agreement=observed, confusion_matrix=confusion, human_label_counts=human_counts,
                balanced_agreement=statistics.mean([confusion[label][label] / count for label, count in human_counts.items() if count]) if n else None,
                cohens_kappa=(observed - chance) / (1 - chance) if n and chance < 1 - 1e-12 else None,
                kappa_reason=None if n and chance < 1 - 1e-12 else "undefined_without_label_variation",
                agreement_cluster_bootstrap=_cluster_interval(records, correct))
    return base


def calibration_report(records, rubric=None, judge=None):
    """Report agreement by task and split; never label a judge 'reliable'.

    Reject duplicates (including renamed records) and development/heldout
    recording leakage. Empty/development-only inputs return a useful incomplete
    report. Heldout samples are necessary for validation but do not themselves
    certify reliability, expert independence, or unbiased sample selection.
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    rubric = validate_rubric(rubric) if rubric is not None else default_rubric()
    items = [validate_calibration_record(record, rubric, judge) for record in records]
    judge_ids = {json.dumps(item["judge"], sort_keys=True) for item in items}
    if len(judge_ids) > 1:
        raise ValueError("select one judge configuration before calibration")
    seen_ids, seen_bindings, split_groups, prediction_bindings, claim_texts = set(), set(), {}, {}, {}
    for item in items:
        if item["id"] in seen_ids:
            raise ValueError("duplicate calibration id")
        seen_ids.add(item["id"])
        output = item["prediction_sha256"]
        kind = item["kind"]
        if kind == "pairwise":
            outputs = tuple(sorted((output, item["other_prediction_sha256"])))
            observation = outputs
        elif kind == "claim" or (kind == "dimension" and item["dimension"] == "factual_correctness"):
            # Rewording an existing claim_id is an error; renaming identical
            # claim text cannot manufacture a second independent observation.
            key = (item["frame_id"], output, item["claim_id"])
            text = " ".join(item["claim_text"].casefold().split())
            if key in claim_texts and claim_texts[key] != text:
                raise ValueError("claim_id refers to different claim text")
            claim_texts[key] = text
            observation = (output, text)
        elif kind == "correctness":
            observation = (output, "overall_correctness")
        else:
            observation = (output, item["dimension"], item.get("fact_id"))
        binding = (item["frame_id"], item["input_sha256"], kind, observation)
        if binding in seen_bindings:
            raise ValueError("duplicate calibration observation; adjudicate human labels into one record")
        seen_bindings.add(binding)
        for digest in (output, item.get("other_prediction_sha256")):
            if digest:
                key = (item["frame_id"], digest)
                if key in prediction_bindings and prediction_bindings[key] != item["input_sha256"]:
                    raise ValueError("same frame/output is bound to different inputs")
                prediction_bindings[key] = item["input_sha256"]
        split_groups.setdefault(item["recording_id"], set()).add(item["split"])
    if any(len(splits) > 1 for splits in split_groups.values()):
        raise ValueError("recording occurs in development and heldout splits")
    splits = {}
    for split in ("development", "heldout"):
        selected = [item for item in items if item["split"] == split]
        splits[split] = {"n": len(selected), "frame_count": len({r["frame_id"] for r in selected}),
                         "recording_count": len({r["recording_id"] for r in selected}),
                         "metrics": {kind: _agreement([item for item in selected if item["kind"] == kind], labels) for kind, labels in _CATEGORICAL.items()}}
        for dimension in _CALIBRATION_DIMENSIONS:
            splits[split]["metrics"][dimension] = _agreement([item for item in selected if item["kind"] == "dimension" and item["dimension"] == dimension])
    heldout = splits["heldout"]
    limits = ["Human verification is an explicit provenance declaration and requires audit of exact output and input bindings.",
              "Rows sharing frames or recordings are correlated; intervals resample recordings, not individual labels.",
              "Selected historical examples and labels do not establish performance on unseen recordings or training independence.",
              "This report does not automatically certify a judge or set scientific acceptance thresholds.",
              "A rubric/GT/input/judge change requires fresh calibration or an explicitly justified compatibility review."]
    if not heldout["n"]:
        limits.append("No heldout human-verified observations are available; development agreement is not validation.")
    elif heldout["recording_count"] < 5:
        limits.append("Fewer than five heldout recordings: uncertainty estimates and generalization evidence are limited.")
    for kind, labels in _CATEGORICAL.items():
        metric = heldout["metrics"][kind]
        if metric["n"] and any(metric["human_label_counts"][label] == 0 for label in labels):
            limits.append(kind + " heldout labels omit at least one category; category-specific reliability remains unknown.")
    return {"schema_version": "mousebench.calibration-report.v1", "rubric_sha256": rubric_hash(rubric),
            "judge": json.loads(next(iter(judge_ids))) if judge_ids else (_judge(judge) if judge else None),
            "status": "provisional_heldout_evidence" if heldout["n"] else "needs_heldout_evidence",
            "reliability_certified": False, "n": len(items), "split_unit": "recording_id", "splits": splits,
            "frame_count": len({r["frame_id"] for r in items}), "recording_count": len(split_groups),
            "coverage_denominator": "submitted_verified_calibration_records_not_full_benchmark",
            "disagreement_ids": [r["id"] for r in items if r["human_label"] != r["judge_label"]],
            "limits": limits}
