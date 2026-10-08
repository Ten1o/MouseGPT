"""Actual request format takes precedence over older imported metadata labels."""
import hashlib

SIX_FIELDS = ('Overall', 'Head', 'Limb', 'Torso', 'Others', 'Keywords')
# Exact system instruction shared by the 30 recovered November-2023 requests.
# It explicitly asks for Description and Keywords. Its original text is kept
# unchanged; existing logged-contract hashes still include the legacy metadata.
TWO_FIELD_SYSTEM_SHA256 = '6952a9cda680083312b49cbb2921fad72784f8e2c8cdda5ff5201dd93b511498'


def expected_output_fields(contract=None):
    contract = contract or {}
    system = contract.get('system_prompt')
    if isinstance(system, str) and hashlib.sha256(system.encode()).hexdigest() == TWO_FIELD_SYSTEM_SHA256:
        return ('Description', 'Keywords')
    fields = contract.get('output_fields')
    if (isinstance(fields, (list, tuple)) and fields and all(isinstance(field, str) and field for field in fields)
            and len(fields) == len(set(fields))):
        return tuple(fields)
    return SIX_FIELDS


def output_format_evidence(contract):
    expected = expected_output_fields(contract)
    declared = tuple((contract or {}).get('output_fields') or SIX_FIELDS)
    return {'expected_fields': list(expected), 'declared_metadata_fields': list(declared),
            'metadata_mismatch': expected != declared,
            'source': 'recognized_original_system_prompt' if expected != declared else 'input_contract',
            'note': 'Assess candidate output organization against the requested fields; six GT sections organize the reference and do not add output-format requirements.'}
