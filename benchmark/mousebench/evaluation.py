"""Strict, offline validation and scoring for versioned MouseGPT assessments.

This module does not establish visual truth or call any model. The application
must verify all assessment hashes against the stored input, reference and answer
before calling score_judgment. All semantic verdicts remain judge-assisted.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import statistics
import unicodedata
from .identity import FRAME_ID

JUDGMENT_SCHEMA_VERSION = "mousebench.judgment.v1"
SCORING_VERSION = "mousebench.factual-f1.v1"
DIAGNOSTICS_VERSION = "mousebench.grounding-diagnostics.v1"
GRADED_JUDGMENT_SCHEMA_VERSION = "mousebench.judgment.v2"
GRADED_SCORING_VERSION = "mousebench.graded-factual-f1.v2"
GRADED_DIAGNOSTICS_VERSION = "mousebench.grounding-diagnostics.v2"
DIAGNOSTIC_INDICATORS = ("any_contradiction", "any_unverifiable",
                         "fully_grounded_nonempty", "complete_required_coverage")
CLAIM_LABELS = {"supported", "contradicted", "unverifiable"}
TRACKS = {"historical", "controlled", "standardized"}
DIMENSIONS = ("factual_precision", "required_fact_recall", "factual_f1",
              "contradiction_rate", "unverifiable_rate", "structure_score",
              "instruction_score")
_HASH = re.compile(r"^[a-fA-F0-9]{64}$")
_FRAME = FRAME_ID
OUTPUT_FIELDS = ("Overall", "Head", "Limb", "Torso", "Others", "Keywords")
GRADED_FIELDS = OUTPUT_FIELDS[:-1]
GRADE_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)
GRADED_FIELD_WEIGHTS = {"Overall": 2.0, "Head": 1.0, "Limb": 1.0,
                        "Torso": 1.0, "Others": 0.5, "Keywords": 1.0}
GRADED_DIMENSIONS = DIMENSIONS + ("major_error_rate",)


def analyze_output(output, expected_fields=OUTPUT_FIELDS):
    """Describe the original JSON and six-field contract without repairing it.

    JSON syntax validity is distinct from the requested object contract. Fences,
    prose and non-finite numbers are not stripped or coerced. Object pairs are
    retained so a duplicate key never silently authorizes its final value.
    This is presentation/format evidence, not a semantic judgment or score.
    """
    result = {"json_valid": False, "contract_valid": False, "parse_error": None, "expected_fields": list(expected_fields),
              "fields": [], "extra_fields": [],
              "mechanical_checks": check_output_structure(output, expected_fields)}
    if not isinstance(output, str):
        result["parse_error"] = {"message": "Output must be a string", "line": None, "column": None}
        return result

    class ObjectPairs(list):
        pass

    def nonfinite(token):
        # Locate the actual numeric token, skipping quoted strings. Do not
        # report a matching word or number inside candidate prose as the error.
        tokens = r'"(?:[^"\\]|\\.)*"|(?P<number>-?Infinity|NaN|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)'
        for match in re.finditer(tokens, output):
            if match.group("number") == token:
                raise json.JSONDecodeError("Non-finite numbers are not permitted", output, match.start())
        raise ValueError("Non-finite numbers are not permitted")

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            nonfinite(token)
        return value

    try:
        parsed = json.loads(output, object_pairs_hook=ObjectPairs,
                            parse_constant=nonfinite, parse_float=finite_float)
    except json.JSONDecodeError as exc:
        result["parse_error"] = {"message": exc.msg, "line": exc.lineno, "column": exc.colno}
        return result
    except (ValueError, RecursionError) as exc:
        # Decoder limits may not expose a source location; never invent one.
        result["parse_error"] = {"message": str(exc), "line": None, "column": None}
        return result
    result["json_valid"] = True
    if not isinstance(parsed, ObjectPairs):
        # The legacy structure checker also supports heading-based text. This
        # strict display view knows the original parsed as a JSON scalar/array;
        # report the actual object-schema defect, not missing text headings.
        result["mechanical_checks"].update(format="json", check="strict_six_field_json" if tuple(expected_fields)==OUTPUT_FIELDS else "strict_requested_field_json",
                                            passed=False, missing_fields=[], unscoped_preamble=False,
                                            errors=["The JSON value must be an object."])
        return result

    def safe_value(value):
        if isinstance(value, ObjectPairs):
            groups = {}
            for key, item in value:
                groups.setdefault(key, []).append(item)
            return {key: safe_value(items[0]) if len(items) == 1 else None
                    for key, items in groups.items()}
        if isinstance(value, list):
            return [safe_value(item) for item in value]
        return value

    values = {}
    for key, value in parsed:
        values.setdefault(key, []).append(value)
    for name in expected_fields:
        entries = values.get(name, [])
        value = safe_value(entries[0]) if len(entries) == 1 else None
        if not entries:
            status = "missing"
        elif len(entries) > 1:
            status = "duplicate"
        elif not isinstance(value, str):
            status = "wrong_type"
        elif not value.strip():
            status = "empty"
        else:
            status = "ok"
        result["fields"].append({"name": name, "value": value, "status": status})
    result["extra_fields"] = [{"name": name, "value": safe_value(items[0]) if len(items) == 1 else None}
                              for name, items in values.items() if name not in expected_fields]
    result["contract_valid"] = not result["extra_fields"] and all(field["status"] == "ok" for field in result["fields"])
    return result


def check_output_structure(output, expected_fields=OUTPUT_FIELDS):
    """Check the original six-section format without repair or semantic scoring.

    Accept either a strict JSON object with exactly six nonempty string fields,
    or six unique anchored text headings with nonempty section bodies. Text
    headings may use Markdown bullets/bold/hash decorations and colon variants.
    Text mode is a section-presence check, not strict JSON schema compliance.
    """
    result = {"check": "six_section_presence" if tuple(expected_fields)==OUTPUT_FIELDS else "requested_section_presence", "passed": False, "format": "unknown",
              "missing_fields": [], "duplicate_fields": [], "empty_fields": [],
              "unexpected_fields": [], "type_errors": [], "errors": [],
              "unscoped_preamble": False}
    if not isinstance(output, str) or not output.strip():
        result.update(format="empty", missing_fields=list(expected_fields), errors=["A nonempty string is required."])
        return result
    if output.lstrip().startswith(("{", "[")):
        duplicates = []

        def pairs(items):
            obj = {}
            for key, value in items:
                if key in obj:
                    duplicates.append(key)
                obj[key] = value
            return obj

        try:
            parsed = json.loads(output, object_pairs_hook=pairs,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON constant")))
        except (ValueError, TypeError, RecursionError) as exc:
            result.update(format="malformed_json", errors=[str(exc)])
            return result
        result["format"] = "json"
        result["check"] = "strict_six_field_json" if tuple(expected_fields)==OUTPUT_FIELDS else "strict_requested_field_json"
        if not isinstance(parsed, dict):
            result["errors"] = ["The JSON value must be an object."]
            return result
        result["duplicate_fields"] = sorted(set(duplicates))
        result["missing_fields"] = [key for key in expected_fields if key not in parsed]
        result["unexpected_fields"] = sorted(set(parsed) - set(expected_fields))
        result["type_errors"] = [key for key in expected_fields if key in parsed and not isinstance(parsed[key], str)]
        result["empty_fields"] = [key for key in expected_fields if isinstance(parsed.get(key), str) and not parsed[key].strip()]
    else:
        result["format"] = "sectioned_text"
        sections, current, preamble = {}, None, []
        names = {key.casefold(): key for key in expected_fields}
        for raw_line in output.splitlines():
            line = re.sub(r"^\s*(?:#{1,6}\s*|(?:[-*]|\d+[.)])\s+)", "", raw_line).strip()
            decorated = line.startswith(("**", "__"))
            if decorated:
                line = line[2:]
            match = re.match(r"^("+'|'.join(re.escape(key) for key in expected_fields)+r")\b(.*)$", line, re.I)
            heading, body = None, None
            if match:
                tail = match.group(2).strip()
                if tail.startswith(("**", "__")):
                    tail = tail[2:].lstrip()
                if tail.startswith((":", "：")):
                    tail = tail[1:].lstrip()
                    if decorated and tail.startswith(("**", "__")):
                        tail = tail[2:].lstrip()
                    heading, body = names[match.group(1).casefold()], tail
                elif not tail:
                    heading, body = names[match.group(1).casefold()], ""
            if heading:
                current = heading
                if heading in sections:
                    result["duplicate_fields"].append(heading)
                sections.setdefault(heading, []).append(body)
            elif current is not None:
                sections[current].append(raw_line)
            else:
                preamble.append(raw_line)
        result["unscoped_preamble"] = bool("".join(preamble).strip())
        result["missing_fields"] = [key for key in expected_fields if key not in sections]
        result["empty_fields"] = [key for key, lines in sections.items() if not "".join(lines).strip()]
        result["duplicate_fields"] = sorted(set(result["duplicate_fields"]))
    result["passed"] = not any(result[key] for key in
                                ("missing_fields", "duplicate_fields", "empty_fields", "unexpected_fields", "type_errors", "errors"))
    return result


def _text(value, name, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a {'possibly empty ' if allow_empty else 'nonempty '}string")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _alias(record, canonical, alias, required=False):
    first, second = record.get(canonical), record.get(alias)
    if first is not None and second is not None and first != second:
        raise ValueError(f"{canonical} and {alias} disagree")
    value = first if first is not None else second
    if required and value is None:
        raise ValueError(f"{canonical} is required")
    return value


def validate_prediction(record, known_frames: set, track="historical"):
    """Return a copied, normalized prediction; reject ambiguous bindings.

    Required: frame_id, run_id and output_text (alias output). Status defaults to
    succeeded. failed, timeout, cancelled and missing records may omit output.
    input_sha256/input_hash is optional for historical records and required for
    controlled/standardized records. Model/run identity is validated upstream.
    Unknown metadata is preserved; dataset rows are never converted into logs.
    """
    if not isinstance(record, dict):
        raise ValueError("prediction must be an object")
    if track not in TRACKS:
        raise ValueError(f"unsupported track: {track}")
    result = copy.deepcopy(record)
    frame_id = _text(record.get("frame_id"), "frame_id")
    if not _FRAME.fullmatch(frame_id) or frame_id not in known_frames:
        raise ValueError(f"unknown or invalid frame_id: {frame_id}")
    _text(record.get("run_id"), "run_id")
    status = record.get("status", "succeeded")
    if status not in {"succeeded", "failed", "timeout", "cancelled", "missing"}:
        raise ValueError("invalid prediction status")
    output = _alias(record, "output_text", "output", status == "succeeded")
    if output is not None:
        _text(output, "output_text", allow_empty=True)
    input_hash = _alias(record, "input_sha256", "input_hash", track != "historical")
    if input_hash is not None:
        result["input_sha256"] = result["input_hash"] = _hash(input_hash, "input_sha256")
    for field in ("prompt_sha256", "prediction_sha256", "request_sha256"):
        if field in record and record[field] is not None:
            result[field] = _hash(record[field], field)
    if record.get("track") is not None and record["track"] != track:
        raise ValueError("prediction track differs from requested track")
    logs = record.get("logs", [])
    if not isinstance(logs, list):
        raise ValueError("logs must be a list")
    for log in logs:
        if not isinstance(log, dict):
            raise ValueError("each log must be an object")
        if log.get("kind") not in {"native_execution", "reconstructed_import", "dataset_record"}:
            raise ValueError("log kind must distinguish native execution, reconstructed import, and dataset record")
        if log.get("kind") == "dataset_record" and log.get("is_execution_log") is True:
            raise ValueError("dataset records cannot be marked as execution logs")
    result.update(output_text=output, output=output, status=status, track=track)
    return result


def _facts(gt_content):
    if not isinstance(gt_content, dict) or gt_content.get("status") != "approved":
        raise ValueError("an approved ground truth is required")
    facts = gt_content.get("facts")
    if not isinstance(facts, list):
        raise ValueError("ground truth facts must be a list")
    all_ids, required_ids = set(), set()
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("each ground truth fact must be an object")
        fact_id = _text(fact.get("id"), "fact.id")
        _text(fact.get("text"), "fact.text")
        if fact_id in all_ids:
            raise ValueError(f"duplicate ground truth fact id: {fact_id}")
        if not isinstance(fact.get("required"), bool):
            raise ValueError("fact.required must be explicitly true or false")
        if "uncertain" in fact and not isinstance(fact["uncertain"], bool):
            raise ValueError("fact.uncertain must be a boolean")
        if fact.get("required") and fact.get("uncertain", False):
            raise ValueError("uncertain facts cannot be required")
        all_ids.add(fact_id)
        if fact["required"]:
            required_ids.add(fact_id)
    return all_ids, required_ids


def _normalized_claim(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def score_judgment(judgment, gt_content, rubric=None):
    """Dispatch by immutable judgment schema; v2 requires its bound rubric."""
    if isinstance(judgment, dict) and judgment.get("schema_version") == GRADED_JUDGMENT_SCHEMA_VERSION:
        return _score_graded_judgment(judgment, gt_content, rubric)
    return _score_binary_judgment(judgment, gt_content)


def prepare_judgment_scorer(rubric):
    """Validate one rubric once for a request while still validating every verdict.

    Capture an independent normalized copy so subsequent edits cannot change the
    scoring policy mid-pass. Public score_judgment retains its full validation.
    """
    from .protocol import validate_rubric, rubric_hash
    normalized = validate_rubric(rubric) if rubric is not None else None
    expected_hash = rubric_hash(normalized) if normalized is not None else None
    def score(judgment, gt_content):
        if isinstance(judgment, dict) and judgment.get("schema_version") == GRADED_JUDGMENT_SCHEMA_VERSION:
            return _score_graded_judgment(judgment, gt_content, normalized, expected_hash)
        return _score_binary_judgment(judgment, gt_content)
    return score


def _score_binary_judgment(judgment, gt_content):
    """Validate a v1 judgment and compute explicit, unweighted factual F1.

    Missing/unknown GT is an error. Approved references with no required facts
    are unevaluable for factual F1 (null, not perfect or zero). Empty claims earn
    precision zero; where required facts exist they also earn recall/F1 zero.
    Exact normalized repeated claims count once; inconsistent repeats fail.
    Required-fact IDs absent from the approved reference fail rather than being
    silently ignored. Upstream is responsible for visual and content bindings.
    """
    if not isinstance(judgment, dict):
        raise ValueError("judgment must be an object")
    required = {"schema_version", "frame_id", "run_id", "protocol_version", "judge",
                "rubric_sha256", "ground_truth_sha256", "prediction_sha256",
                "claims", "supported_fact_ids", "structure_score", "instruction_score"}
    optional = {"input_sha256", "input_hash", "track", "recording_id", "assessment_id",
                "created_at", "notes", "structure_checks", "calibration_status"}
    unknown = set(judgment) - required - optional
    if unknown:
        raise ValueError(f"unknown judgment fields: {', '.join(sorted(unknown))}")
    missing = required - set(judgment)
    if missing:
        raise ValueError(f"missing judgment fields: {', '.join(sorted(missing))}")
    if judgment["schema_version"] != JUDGMENT_SCHEMA_VERSION:
        raise ValueError("unsupported judgment schema_version")
    for field in ("frame_id", "run_id", "protocol_version"):
        _text(judgment[field], field)
    if not _FRAME.fullmatch(judgment["frame_id"]):
        raise ValueError("invalid frame_id")
    judge = judgment["judge"]
    if not isinstance(judge, dict) or set(judge) != {"provider", "model", "version"}:
        raise ValueError("judge must contain exactly provider, model and version")
    for field in ("provider", "model", "version"):
        _text(judge[field], f"judge.{field}")
    bindings = {name: _hash(judgment[name], name) for name in
                ("rubric_sha256", "ground_truth_sha256", "prediction_sha256")}
    bindings["input_sha256"] = _hash(_alias(judgment, "input_sha256", "input_hash", True), "input_sha256")
    track = judgment.get("track", "historical")
    if track not in TRACKS:
        raise ValueError("unsupported judgment track")
    all_ids, required_ids = _facts(gt_content)
    covered = judgment["supported_fact_ids"]
    if not isinstance(covered, list) or any(not isinstance(x, str) for x in covered):
        raise ValueError("supported_fact_ids must be a list of strings")
    if len(covered) != len(set(covered)):
        raise ValueError("supported_fact_ids must not contain duplicates")
    if set(covered) - required_ids:
        raise ValueError("supported_fact_ids must refer only to approved required facts")
    claims = judgment["claims"]
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    distinct = {}
    mapped = any(isinstance(claim, dict) and "fact_ids" in claim for claim in claims)
    supported_mappings = set()
    for claim in claims:
        if not isinstance(claim, dict) or not {"text", "label"} <= set(claim):
            raise ValueError("each claim requires text and label")
        if set(claim) - {"text", "label", "evidence", "fact_ids"}:
            raise ValueError("unknown claim fields")
        normalized = _normalized_claim(_text(claim["text"], "claim.text"))
        if not isinstance(claim["label"], str) or claim["label"] not in CLAIM_LABELS:
            raise ValueError("invalid claim label")
        if "evidence" in claim:
            _text(claim["evidence"], "claim.evidence", allow_empty=True)
        if mapped and "fact_ids" not in claim:
            raise ValueError("all claims in a mapped judgment must provide fact_ids")
        if "fact_ids" in claim:
            ids = claim["fact_ids"]
            if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                raise ValueError("claim.fact_ids must be a list of strings")
            if len(ids) != len(set(ids)):
                raise ValueError("claim.fact_ids must not contain duplicates")
            if set(ids) - all_ids:
                raise ValueError("claim.fact_ids contains an unknown approved fact")
            if claim["label"] == "supported":
                supported_mappings.update(set(ids) & required_ids)
        if normalized in distinct and distinct[normalized] != claim["label"]:
            raise ValueError("duplicate claim has inconsistent labels")
        distinct[normalized] = claim["label"]
    if mapped and supported_mappings != set(covered):
        raise ValueError("supported_fact_ids must equal required facts mapped by supported claims")
    counts = {label: sum(value == label for value in distinct.values()) for label in CLAIM_LABELS}
    if covered and counts["supported"] == 0:
        raise ValueError("required fact coverage needs at least one supported claim")
    total = len(distinct)
    precision = counts["supported"] / total if total else 0.0
    recall = len(covered) / len(required_ids) if required_ids else None
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if recall is not None else None
    result = {
        "scoring_version": SCORING_VERSION, "protocol_version": judgment["protocol_version"],
        "frame_id": judgment["frame_id"], "run_id": judgment["run_id"], "track": track,
        "recording_id": judgment.get("recording_id") or judgment["frame_id"].rsplit("_", 1)[0],
        "status": "scored" if required_ids else "unscorable_reference", "evaluable": bool(required_ids),
        "reason": None if required_ids else "no_required_facts",
        "judge": copy.deepcopy(judge), **bindings,
        "factual_precision": precision, "required_fact_recall": recall, "factual_f1": f1,
        "contradiction_rate": counts["contradicted"] / total if total else 0.0,
        "unverifiable_rate": counts["unverifiable"] / total if total else 0.0,
        "structure_score": _unit(judgment["structure_score"], "structure_score"),
        "instruction_score": _unit(judgment["instruction_score"], "instruction_score"),
        "denominators": {"claims_submitted": len(claims), "distinct_asserted_claims": total,
                         "supported_claims": counts["supported"], "contradicted_claims": counts["contradicted"],
                         "unverifiable_claims": counts["unverifiable"], "required_facts": len(required_ids),
                         "covered_required_facts": len(covered)},
        "supported_fact_ids": sorted(covered), "missing_required_fact_ids": sorted(required_ids - set(covered)),
        "calibration_status": judgment.get("calibration_status", "unverified"),
    }
    result["grounding_diagnostics"] = _grounding_diagnostics(gt_content, required_ids, set(covered), counts, total)
    return result


def _quarter(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in GRADE_VALUES:
        raise ValueError(name + " must be one of 0, 0.25, 0.5, 0.75 or 1")
    return float(value)


def _id_list(value, name):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(name + " must be a list of nonempty IDs")
    if len(value) != len(set(value)):
        raise ValueError(name + " must not contain duplicate IDs")
    return value


def _graded_policy(judgment, rubric, prepared_hash=None):
    if not isinstance(rubric, dict) or not isinstance(rubric.get("scoring"), dict):
        raise ValueError("A bound rubric with the graded scoring policy is required")
    spec = rubric["scoring"]
    if spec.get("version") != GRADED_SCORING_VERSION:
        raise ValueError("Rubric scoring.version does not select graded factual F1 v2")
    weights = spec.get("field_weights")
    if not isinstance(weights, dict) or set(weights) != set(GRADED_FIELD_WEIGHTS):
        raise ValueError("Graded scoring requires all six configured field weights")
    for field, expected in GRADED_FIELD_WEIGHTS.items():
        value = weights[field]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or value != expected):
            raise ValueError("Unsupported graded field weight for " + field)
    fallback = spec.get("unknown_gt_weight", 1)
    if isinstance(fallback, bool) or not isinstance(fallback, (int, float)) or fallback != 1:
        raise ValueError("Graded unassigned GT field weight must be 1")
    # Lazy import avoids the protocol module's imports of this module at startup.
    from .protocol import rubric_hash
    if _hash(judgment.get("rubric_sha256"), "rubric_sha256") != (prepared_hash if prepared_hash is not None else rubric_hash(rubric)):
        raise ValueError("Judgment rubric_sha256 does not match the supplied rubric")
    if judgment.get("protocol_version") != rubric.get("id"):
        raise ValueError("Judgment protocol_version does not match the supplied rubric")
    return dict(GRADED_FIELD_WEIGHTS)


def _score_graded_judgment(judgment, gt_content, rubric, prepared_hash=None):
    """Two-sided graded factual F1; error labels are diagnostics, not deductions.

    Candidate correctness and required-fact coverage are different quantities.
    Each side uses fixed semantic-field weights from the bound rubric. Repeated
    normalized assertions count once; semantic paraphrase deduplication remains
    the judge's responsibility. No source text, GT or judgment is modified.
    """
    required = {"schema_version", "frame_id", "run_id", "protocol_version", "judge",
                "rubric_sha256", "ground_truth_sha256", "prediction_sha256",
                "claims", "fact_coverage", "structure_score", "instruction_score"}
    optional = {"input_sha256", "input_hash", "track", "recording_id", "assessment_id",
                "created_at", "notes", "structure_checks", "calibration_status"}
    if set(judgment) - required - optional:
        raise ValueError("unknown graded judgment fields: " + ", ".join(sorted(set(judgment) - required - optional)))
    if required - set(judgment):
        raise ValueError("missing graded judgment fields: " + ", ".join(sorted(required - set(judgment))))
    weights = _graded_policy(judgment, rubric, prepared_hash)
    for field in ("frame_id", "run_id", "protocol_version"):
        _text(judgment[field], field)
    if not _FRAME.fullmatch(judgment["frame_id"]):
        raise ValueError("invalid frame_id")
    judge = judgment["judge"]
    if not isinstance(judge, dict) or set(judge) != {"provider", "model", "version"}:
        raise ValueError("judge must contain exactly provider, model and version")
    for field in ("provider", "model", "version"):
        _text(judge[field], "judge." + field)
    bindings = {name: _hash(judgment[name], name) for name in
                ("rubric_sha256", "ground_truth_sha256", "prediction_sha256")}
    bindings["input_sha256"] = _hash(_alias(judgment, "input_sha256", "input_hash", True), "input_sha256")
    track = judgment.get("track", "historical")
    if not isinstance(track, str) or track not in TRACKS:
        raise ValueError("unsupported judgment track")
    if judgment.get("recording_id") is not None:
        _text(judgment["recording_id"], "recording_id")
    all_ids, required_ids = _facts(gt_content)
    fact_lookup = {fact["id"]: fact for fact in gt_content["facts"]}
    claims = judgment["claims"]
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    claim_fields = {"id", "text", "field", "correctness", "evidence", "error_type", "severity", "fact_ids"}
    distinct, aliases, claim_by_id = {}, {}, {}
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != claim_fields:
            raise ValueError("Each graded claim must contain exactly id, text, field, correctness, evidence, error_type, severity and fact_ids")
        cid = _text(claim["id"], "claim.id")
        if cid in aliases:
            raise ValueError("duplicate claim id: " + cid)
        normalized = _normalized_claim(_text(claim["text"], "claim.text"))
        field = claim["field"]
        if not isinstance(field, str) or field not in GRADED_FIELDS:
            raise ValueError("claim.field must identify its semantic home; Keywords is not a candidate claim field")
        correctness = _quarter(claim["correctness"], "claim.correctness")
        evidence = _text(claim["evidence"], "claim.evidence")
        error, severity = claim["error_type"], claim["severity"]
        if not isinstance(error, str) or error not in {"none", "contradicted", "unverifiable", "mixed"}:
            raise ValueError("invalid claim.error_type")
        if not isinstance(severity, str) or severity not in {"none", "minor", "major"}:
            raise ValueError("invalid claim.severity")
        if correctness == 1:
            if error != "none" or severity != "none":
                raise ValueError("Fully correct claims must have no error type or severity")
        elif error == "none" or severity == "none":
            raise ValueError("Claims below full correctness require an error type and severity")
        ids = _id_list(claim["fact_ids"], "claim.fact_ids")
        if set(ids) - all_ids:
            raise ValueError("claim.fact_ids contains an unknown approved fact")
        if normalized in distinct:
            canonical = distinct[normalized]
            if (canonical["field"], canonical["correctness"], canonical["error_type"], canonical["severity"]) != (field, correctness, error, severity):
                raise ValueError("duplicate claim has inconsistent fields or grades")
            canonical["aliases"].append(cid)
            canonical["fact_ids"] = sorted(set(canonical["fact_ids"]) | set(ids))
            if evidence != canonical["evidence"] and evidence not in canonical["additional_evidence"]:
                canonical["additional_evidence"].append(evidence)
        else:
            canonical = {"id": cid, "text": claim["text"], "field": field,
                         "correctness": correctness, "weight": weights[field],
                         "weighted_correctness_credit": weights[field] * correctness,
                         "evidence": evidence, "additional_evidence": [],
                         "error_type": error, "severity": severity, "fact_ids": sorted(ids), "aliases": []}
            distinct[normalized] = canonical
            claim_by_id[cid] = canonical
        aliases[cid] = canonical["id"]
    canonical_claims = list(distinct.values())
    coverage = judgment["fact_coverage"]
    if not isinstance(coverage, list):
        raise ValueError("fact_coverage must be a list")
    coverage_by_id = {}
    unassigned = []
    for item in coverage:
        if not isinstance(item, dict) or set(item) != {"fact_id", "coverage", "claim_ids", "evidence"}:
            raise ValueError("Each fact coverage row must contain exactly fact_id, coverage, claim_ids and evidence")
        fid = _text(item["fact_id"], "fact_coverage.fact_id")
        if fid not in required_ids:
            raise ValueError("fact_coverage must refer only to required approved facts")
        if fid in coverage_by_id:
            raise ValueError("duplicate required fact coverage: " + fid)
        credit = _quarter(item["coverage"], "fact_coverage.coverage")
        evidence = _text(item["evidence"], "fact_coverage.evidence")
        linked = _id_list(item["claim_ids"], "fact_coverage.claim_ids")
        if set(linked) - set(aliases):
            raise ValueError("fact_coverage contains an unknown claim ID")
        canonical_ids = sorted({aliases[cid] for cid in linked})
        if any(fid not in claim_by_id[cid]["fact_ids"] for cid in canonical_ids):
            raise ValueError("Coverage links must map to that approved fact in claim.fact_ids")
        if credit > 0 and not any(claim_by_id[cid]["correctness"] > 0 for cid in canonical_ids):
            raise ValueError("Positive fact coverage requires a linked claim with nonzero correctness")
        field = fact_lookup[fid].get("field")
        assigned = isinstance(field, str) and field in weights
        weight = weights[field] if assigned else 1.0
        if not assigned:
            unassigned.append(fid)
        coverage_by_id[fid] = {"fact_id": fid, "coverage": credit, "claim_ids": canonical_ids,
                               "evidence": evidence, "field": copy.deepcopy(field), "weight": weight,
                               "weighted_coverage_credit": credit * weight,
                               "unassigned_field_fallback": not assigned}
    if set(coverage_by_id) != required_ids:
        raise ValueError("fact_coverage must assess every required fact exactly once")
    total = len(canonical_claims)
    claim_weight = sum(claim["weight"] for claim in canonical_claims)
    correctness_credit = sum(claim["weighted_correctness_credit"] for claim in canonical_claims)
    required_weight = sum(item["weight"] for item in coverage_by_id.values())
    coverage_credit = sum(item["weighted_coverage_credit"] for item in coverage_by_id.values())
    precision = correctness_credit / claim_weight if claim_weight else 0.0
    recall = coverage_credit / required_weight if required_weight else None
    f1 = (2 * precision * recall / (precision + recall) if precision + recall else 0.0) if recall is not None else None
    full = {fid for fid, item in coverage_by_id.items() if item["coverage"] == 1}
    partial = {fid for fid, item in coverage_by_id.items() if 0 < item["coverage"] < 1}
    contradicted = sum(claim["error_type"] in {"contradicted", "mixed"} for claim in canonical_claims)
    unverifiable = sum(claim["error_type"] in {"unverifiable", "mixed"} for claim in canonical_claims)
    major = sum(claim["severity"] == "major" for claim in canonical_claims)
    full_claims = sum(claim["correctness"] == 1 for claim in canonical_claims)
    partial_claims = sum(0 < claim["correctness"] < 1 for claim in canonical_claims)
    result = {
        "scoring_version": GRADED_SCORING_VERSION, "protocol_version": judgment["protocol_version"],
        "frame_id": judgment["frame_id"], "run_id": judgment["run_id"], "track": track,
        "recording_id": judgment.get("recording_id") or judgment["frame_id"].rsplit("_", 1)[0],
        "status": "scored" if required_ids else "unscorable_reference", "evaluable": bool(required_ids),
        "reason": None if required_ids else "no_required_facts", "judge": copy.deepcopy(judge), **bindings,
        "factual_precision": precision, "required_fact_recall": recall, "factual_f1": f1,
        "contradiction_rate": contradicted / total if total else 0.0,
        "unverifiable_rate": unverifiable / total if total else 0.0,
        "major_error_rate": major / total if total else 0.0, "any_major_error": major > 0,
        "structure_score": _quarter(judgment["structure_score"], "structure_score"),
        "instruction_score": _quarter(judgment["instruction_score"], "instruction_score"),
        "supported_fact_ids": sorted(full), "partially_covered_fact_ids": sorted(partial),
        "missing_required_fact_ids": sorted(required_ids - full - partial),
        "denominators": {"claims_submitted": len(claims), "distinct_asserted_claims": total,
                         "supported_claims": full_claims, "fully_correct_claims": full_claims,
                         "partially_correct_claims": partial_claims, "zero_credit_claims": total - full_claims - partial_claims,
                         "contradicted_claims": contradicted, "unverifiable_claims": unverifiable, "major_error_claims": major,
                         "claim_weight": claim_weight, "weighted_correctness_credit": correctness_credit,
                         "required_facts": len(required_ids), "covered_required_facts": len(full),
                         "partially_covered_required_facts": len(partial), "required_fact_weight": required_weight,
                         "weighted_coverage_credit": coverage_credit},
        "graded_details": {"field_weights": weights, "grade_values": list(GRADE_VALUES),
                           "claims": copy.deepcopy(canonical_claims),
                           "fact_coverage": [coverage_by_id[fid] for fid in sorted(coverage_by_id)],
                           "unassigned_required_fact_ids": sorted(unassigned), "unassigned_field_weight": 1.0,
                           "diagnostic_rates": "Unweighted distinct-claim incidence. Mixed errors count in both contradiction and unverifiable rates; these rates may overlap and never add deductions."},
        "calibration_status": judgment.get("calibration_status", "unverified"),
    }
    result["grounding_diagnostics"] = _graded_grounding_diagnostics(gt_content, coverage_by_id,
        contradicted, unverifiable, major, full_claims, total)
    return result


def _graded_grounding_diagnostics(gt_content, coverage, contradicted, unverifiable, major, full_claims, total):
    sections = {field: {"covered": 0.0, "required": 0, "recall": None} for field in OUTPUT_FIELDS}
    unassigned_required, unassigned_covered = 0, 0.0
    for fact in gt_content["facts"]:
        if fact["id"] not in coverage:
            continue
        credit, field = coverage[fact["id"]]["coverage"], fact.get("field")
        if not isinstance(field, str) or field not in sections:
            unassigned_required += 1
            unassigned_covered += credit
        else:
            sections[field]["required"] += 1
            sections[field]["covered"] += credit
    for section in sections.values():
        if section["required"]:
            section["recall"] = section["covered"] / section["required"]
    return {"version": GRADED_DIAGNOSTICS_VERSION, "required_fact_recall_by_section": sections,
            "unassigned_required_facts": unassigned_required, "unassigned_covered_required_facts": unassigned_covered,
            "any_contradiction": contradicted > 0, "any_unverifiable": unverifiable > 0, "any_major_error": major > 0,
            "fully_grounded_nonempty": total > 0 and full_claims == total,
            "complete_required_coverage": all(item["coverage"] == 1 for item in coverage.values()) if coverage else None}


def _grounding_diagnostics(gt_content, required_ids, covered, counts, total):
    """Descriptive diagnostics; never alter the v1 F1 or introduce error weights."""
    sections = {field: {"covered": 0, "required": 0, "recall": None} for field in OUTPUT_FIELDS}
    unassigned_required = unassigned_covered = 0
    for fact in gt_content["facts"]:
        if fact["id"] not in required_ids:
            continue
        hit = int(fact["id"] in covered)
        field = fact.get("field")
        if not isinstance(field, str) or field not in sections:
            # Older accepted references may lack field metadata. Never guess it.
            unassigned_required += 1
            unassigned_covered += hit
        else:
            sections[field]["required"] += 1
            sections[field]["covered"] += hit
    for section in sections.values():
        if section["required"]:
            section["recall"] = section["covered"] / section["required"]
    return {"version": DIAGNOSTICS_VERSION, "required_fact_recall_by_section": sections,
            "unassigned_required_facts": unassigned_required, "unassigned_covered_required_facts": unassigned_covered,
            "any_contradiction": counts["contradicted"] > 0,
            "any_unverifiable": counts["unverifiable"] > 0,
            "fully_grounded_nonempty": total > 0 and counts["contradicted"] == 0 and counts["unverifiable"] == 0,
            "complete_required_coverage": covered == required_ids if required_ids else None}


def _count(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(label + " must be a nonnegative integer")
    return value


def _validated_diagnostics(value):
    if not isinstance(value, dict) or value.get("version") not in {DIAGNOSTICS_VERSION, GRADED_DIAGNOSTICS_VERSION}:
        raise ValueError("unsupported grounding diagnostics version")
    graded = value["version"] == GRADED_DIAGNOSTICS_VERSION
    sections = value.get("required_fact_recall_by_section")
    if not isinstance(sections, dict) or set(sections) != set(OUTPUT_FIELDS):
        raise ValueError("grounding diagnostics must contain all six section denominators")
    for field, section in sections.items():
        if not isinstance(section, dict):
            raise ValueError("section diagnostics must be an object")
        required = _count(section.get("required"), field + ".required")
        covered = (_coverage_mass(section.get("covered"), field + ".covered") if graded
                   else _count(section.get("covered"), field + ".covered"))
        if covered > required:
            raise ValueError("covered facts cannot exceed required facts")
        recall = section.get("recall")
        if required:
            if not math.isclose(_unit(recall, field + ".recall"), covered / required, abs_tol=1e-12):
                raise ValueError("section recall disagrees with its numerator and denominator")
        elif recall is not None:
            raise ValueError("section with no required facts must have null recall")
    for name in DIAGNOSTIC_INDICATORS + (("any_major_error",) if graded else ()):
        if name == "complete_required_coverage" and value.get(name) is None:
            continue
        if not isinstance(value.get(name), bool):
            raise ValueError(name + " must be boolean")
    required = _count(value.get("unassigned_required_facts"), "unassigned_required_facts")
    covered = (_coverage_mass(value.get("unassigned_covered_required_facts"), "unassigned_covered_required_facts") if graded
               else _count(value.get("unassigned_covered_required_facts"), "unassigned_covered_required_facts"))
    if covered > required:
        raise ValueError("unassigned covered facts cannot exceed unassigned required facts")
    return value


def _coverage_mass(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or not math.isfinite(value * 4)
            or value * 4 != int(value * 4)):
        raise ValueError(name + " must be a nonnegative quarter-step coverage sum")
    return float(value)


def _recording_macro(pairs):
    recordings = {}
    for record, value in pairs:
        identity = record.get("recording_id") or record["frame_id"].rsplit("_", 1)[0]
        recordings.setdefault(identity, []).append(value)
    return (statistics.mean(statistics.mean(values) for values in recordings.values()) if recordings else None,
            len(recordings))


def _diagnostic_summary(records, scoring_version=None):
    """Aggregate known diagnostics with explicit legacy/NA coverage.

    Section micro recall pools facts; frame macro equally weights applicable
    frames; recording macro first averages those frame recalls per recording.
    Legacy scores lacking diagnostics remain missing, never error-free zeros.
    """
    eligible = [record for record in records if record.get("evaluable") is True]
    graded = scoring_version == GRADED_SCORING_VERSION or any(record.get("scoring_version") == GRADED_SCORING_VERSION for record in records)
    version = GRADED_DIAGNOSTICS_VERSION if graded else DIAGNOSTICS_VERSION
    available = []
    for record in eligible:
        if record.get("grounding_diagnostics") is not None:
            diagnostics = _validated_diagnostics(record["grounding_diagnostics"])
            if diagnostics["version"] != version:
                raise ValueError("Grounding diagnostics version differs from scoring version")
            available.append((record, diagnostics))
    result = {"version": version, "eligible_frame_count": len(eligible),
              "diagnosed_frame_count": len(available), "missing_diagnostics_count": len(eligible) - len(available),
              "coverage": len(available) / len(eligible) if eligible else None,
              "coverage_denominator": "eligible_frames_in_this_summary",
              "frame_indicators": {}, "sections": {},
              "unassigned_required_facts": sum(value["unassigned_required_facts"] for _, value in available),
              "unassigned_covered_required_facts": sum(value["unassigned_covered_required_facts"] for _, value in available)}
    for name in DIAGNOSTIC_INDICATORS + (("any_major_error",) if graded else ()):
        pairs = [(record, int(value[name])) for record, value in available if value[name] is not None]
        macro, groups = _recording_macro(pairs)
        result["frame_indicators"][name] = {"positive_count": sum(value for _, value in pairs), "denominator": len(pairs),
            "rate": statistics.mean(value for _, value in pairs) if pairs else None,
            "recording_macro_rate": macro, "recording_count": groups}
    for field in OUTPUT_FIELDS:
        totals = [value["required_fact_recall_by_section"][field] for _, value in available]
        covered = sum(value["covered"] for value in totals)
        required = sum(value["required"] for value in totals)
        pairs = [(record, value["required_fact_recall_by_section"][field]["recall"]) for record, value in available
                 if value["required_fact_recall_by_section"][field]["required"] > 0]
        macro, groups = _recording_macro(pairs)
        result["sections"][field] = {"covered_required_facts": covered, "required_facts": required,
            "micro_recall": covered / required if required else None,
            "frame_macro_recall": statistics.mean(value for _, value in pairs) if pairs else None,
            "recording_macro_recall": macro, "applicable_frame_count": len(pairs), "applicable_recording_count": groups,
            "not_applicable_frame_count": len(available) - len(pairs)}
    return result


def _means(records, dimensions=DIMENSIONS):
    return {dimension: statistics.mean(values) if values else None
            for dimension in dimensions
            for values in [[r[dimension] for r in records if r.get("evaluable") is True and r.get(dimension) is not None]]}


def _quantile(sorted_values, q):
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def _paired_bootstrap(left, right, common):
    clusters = {}
    for frame in common:
        a, b = left[frame], right[frame]
        if a.get("recording_id") != b.get("recording_id"):
            raise ValueError(f"inconsistent recording identity for {frame}")
        delta = a["factual_f1"] - b["factual_f1"]
        group = a.get("recording_id")
        if not group:
            raise ValueError("recording_id is required for clustered comparisons")
        clusters.setdefault(group, []).append(delta)
    estimate = statistics.mean([d for group in clusters.values() for d in group]) if common else None
    base = {"metric": "factual_f1", "delta_a_minus_b": estimate,
            "cluster_unit": "recording_id", "cluster_count": len(clusters),
            "confidence_level": 0.95, "replicates": 0, "seed": 395, "interval": None}
    if len(clusters) < 2:
        base["reason"] = "at_least_two_recording_clusters_required"
        return base
    summaries = [(sum(values), len(values)) for _, values in sorted(clusters.items())]
    rng, samples = random.Random(395), []
    for _ in range(1000):
        chosen = [rng.choice(summaries) for _ in summaries]
        samples.append(sum(x[0] for x in chosen) / sum(x[1] for x in chosen))
    samples.sort()
    base.update(replicates=1000, interval=[_quantile(samples, .025), _quantile(samples, .975)], reason=None)
    return base


def aggregate_scorecards(records: list):
    """Summarize one protocol/track/scoring version without imputing missing work.

    Input: flat score_judgment results, optionally unscored records with the same
    identity/version fields and evaluable=False. Repeated run/frame rows fail.
    Coverage is relative to observed frame union, NOT the full benchmark size.
    Each pair is compared only on its common evaluable frames; a separate global
    intersection supports a single comparable table. No composite is invented.
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    if not records:
        return {"protocol_version": None, "scoring_version": SCORING_VERSION,
                "track": None, "runs": {}, "common_frame_ids": [], "comparisons": [],
                "observed_frame_count": 0, "coverage_denominator": "observed_frame_union"}
    versions, runs, universe = set(), {}, set()
    frame_bindings, judge_configs = {}, set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each scorecard must be an object")
        key = tuple(_text(record.get(name), name) for name in ("protocol_version", "scoring_version", "track"))
        versions.add(key)
        if record["scoring_version"] not in {SCORING_VERSION, GRADED_SCORING_VERSION} or record["track"] not in TRACKS:
            raise ValueError("unsupported scoring version or track")
        run, frame = _text(record.get("run_id"), "run_id"), _text(record.get("frame_id"), "frame_id")
        if frame in runs.setdefault(run, {}):
            raise ValueError(f"duplicate scorecard for {run}/{frame}; choose a judge run explicitly")
        if not isinstance(record.get("evaluable"), bool):
            raise ValueError("evaluable must be a boolean")
        if record["evaluable"]:
            for dimension in (GRADED_DIMENSIONS if record["scoring_version"] == GRADED_SCORING_VERSION else DIMENSIONS):
                _unit(record.get(dimension), dimension)
            if record["scoring_version"] == GRADED_SCORING_VERSION and not isinstance(record.get("any_major_error"), bool):
                raise ValueError("any_major_error must be boolean for graded scorecards")
            # A common protocol label cannot make different input/reference
            # versions or different judge configurations comparable.
            binding = tuple(record.get(name) for name in ("input_sha256", "ground_truth_sha256"))
            if any(value is None for value in binding):
                raise ValueError("evaluable scorecards require input and ground truth hashes")
            if frame in frame_bindings and frame_bindings[frame] != binding:
                raise ValueError(f"different input/reference bindings for {frame}")
            frame_bindings[frame] = binding
            judge = record.get("judge", {})
            if not isinstance(judge, dict):
                raise ValueError("judge must be an object")
            config = tuple(judge.get(name) for name in ("provider", "model", "version")) + (record.get("rubric_sha256"),)
            if any(not isinstance(value, str) or not value for value in config):
                raise ValueError("evaluable scorecards require judge identity/version and rubric hash")
            judge_configs.add(config)
        runs[run][frame] = record
        universe.add(frame)
    if len(versions) != 1:
        raise ValueError("cannot aggregate different protocol, track or scoring versions")
    if len(judge_configs) > 1:
        raise ValueError("choose one judge configuration and rubric before aggregation")
    protocol, scoring, track = next(iter(versions))
    dimensions = GRADED_DIMENSIONS if scoring == GRADED_SCORING_VERSION else DIMENSIONS
    eligible = {run: {frame: card for frame, card in cards.items() if card["evaluable"]}
                for run, cards in runs.items()}
    common = sorted(set.intersection(*(set(cards) for cards in eligible.values())))
    result = {"protocol_version": protocol, "scoring_version": scoring, "track": track,
              "coverage_denominator": "observed_frame_union", "observed_frame_count": len(universe),
              "common_frame_ids": common, "runs": {}, "comparisons": []}
    for run, cards in sorted(runs.items()):
        ready = eligible[run]
        by_recording = {}
        for card in ready.values():
            group = card.get("recording_id") or card["frame_id"].rsplit("_", 1)[0]
            by_recording.setdefault(group, []).append(card)
        recording_means = [_means(group, dimensions) for group in by_recording.values()]
        result["runs"][run] = {
            "observed_count": len(cards), "evaluable_count": len(ready),
            "unevaluable_count": len(cards) - len(ready),
            "missing_from_observed_union": len(universe - set(cards)),
            "evaluable_coverage": len(ready) / len(universe),
            "dimension_means": _means(list(cards.values()), dimensions),
            "common_frame_dimension_means": _means([cards[frame] for frame in common], dimensions),
            "recording_balanced_means": {dimension: statistics.mean([m[dimension] for m in recording_means]) if recording_means else None for dimension in dimensions},
            "recording_count": len(by_recording),
            "grounding_diagnostics": _diagnostic_summary(list(cards.values()), scoring),
            "common_frame_grounding_diagnostics": _diagnostic_summary([cards[frame] for frame in common], scoring),
        }
    labels = sorted(runs)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            pair_common = sorted(set(eligible[a]) & set(eligible[b]))
            result["comparisons"].append({"run_a": a, "run_b": b, "common_frame_ids": pair_common,
                                          "common_frame_count": len(pair_common),
                                          "paired_cluster_bootstrap": _paired_bootstrap(eligible[a], eligible[b], pair_common)})
    return result
