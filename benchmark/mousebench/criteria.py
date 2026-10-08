"""Evidence-backed analytic criteria, reported separately from numerical scores.

These checks expose reasons for an assessment; neither a checklist nor its met
rate establishes validity. Missing legacy assessments are not passing results.
"""
from __future__ import annotations

import copy
import re

VERDICTS = ("met", "unmet", "not_assessable")
_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + " must be a nonempty string")
    return value.strip()


def validate_criteria(spec):
    """Return normalized definitions; allow an empty list for legacy rubrics."""
    if not isinstance(spec, list):
        raise ValueError("analytic_criteria must be a list")
    result = []
    seen = set()
    required = {"id", "label", "definition", "allow_not_assessable", "group"}
    for raw in spec:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("each criterion must contain exactly id, label, definition, allow_not_assessable and group")
        criterion = copy.deepcopy(raw)
        for key in ("id", "label", "definition", "group"):
            criterion[key] = _text(criterion[key], "criterion." + key)
        if not _ID.fullmatch(criterion["id"]):
            raise ValueError("criterion.id must be a lowercase identifier")
        if criterion["id"] in seen:
            raise ValueError("duplicate analytic criterion id")
        if not isinstance(criterion["allow_not_assessable"], bool):
            raise ValueError("allow_not_assessable must be explicitly boolean")
        seen.add(criterion["id"])
        result.append(criterion)
    return result


def validate_assessments(assessments, spec):
    """Require a verdict and concrete evidence for every configured criterion.

    There is no score, weight, default pass, or inferred assessment. An empty
    definition list permits only an empty mapping, not arbitrary unseen checks.
    """
    definitions = validate_criteria(spec)
    if not isinstance(assessments, dict):
        raise ValueError("criterion assessments must be an object keyed by criterion id")
    expected = {criterion["id"] for criterion in definitions}
    if set(assessments) != expected:
        missing, unknown = sorted(expected - set(assessments)), sorted(str(key) for key in set(assessments) - expected)
        raise ValueError("criterion assessments must match all rubric ids; missing=" + str(missing) + "; unknown=" + str(unknown))
    result = {}
    for criterion in definitions:
        identity = criterion["id"]
        value = assessments[identity]
        if not isinstance(value, dict) or set(value) != {"verdict", "evidence"}:
            raise ValueError(identity + " requires exactly verdict and evidence")
        verdict = value["verdict"]
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            raise ValueError(identity + " has an unsupported verdict")
        if verdict == "not_assessable" and not criterion["allow_not_assessable"]:
            raise ValueError(identity + " does not permit not_assessable")
        result[identity] = {"verdict": verdict, "evidence": _text(value["evidence"], identity + ".evidence")}
    return result


def summarize_assessments(records, spec):
    """Summarize one fixed criterion set; each row is a mapping or missing None.

    Met rate = met / (met + unmet), never including NA or missing rows. Callers
    must restrict rows to one frozen rubric and a compatible judge configuration.
    The result deliberately supplies no aggregate checklist score or pass label.
    """
    if not isinstance(records, list):
        raise ValueError("assessment records must be a list")
    definitions = validate_criteria(spec)
    available = [validate_assessments(record, definitions) for record in records if record is not None]
    missing = len(records) - len(available)
    result = {"record_count": len(records), "assessed_record_count": len(available),
              "missing_record_count": missing,
              "assessment_coverage": len(available) / len(records) if records else None,
              "criteria": {},
              "note": "Criterion-specific descriptive rates only. NA and missing records are excluded from the applicable denominator; no combined score or scientific validity claim."}
    for criterion in definitions:
        identity = criterion["id"]
        counts = {verdict: sum(record[identity]["verdict"] == verdict for record in available) for verdict in VERDICTS}
        applicable = counts["met"] + counts["unmet"]
        result["criteria"][identity] = {"label": criterion["label"], "group": criterion["group"], **counts,
                                       "missing": missing, "applicable_count": applicable,
                                       "met_rate": counts["met"] / applicable if applicable else None}
    return result
