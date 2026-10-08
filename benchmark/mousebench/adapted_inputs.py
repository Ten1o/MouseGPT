"""Versioned benchmark instructions derived from verified, frozen source inputs.

An adapted request is a new controlled input, never an archived execution log.
Only the system message and declared output fields are replaced. The original
per-frame text, speeds, two image objects and their ordering remain unchanged.
"""
import copy
import hashlib

from .output_format import expected_output_fields, SIX_FIELDS
from .recorded_inputs import (ADAPTED_PROTOCOL, PROTOCOL, STRUCTURED_PROTOCOLS,
                              RecordedInputError, verify_input_images)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _pinned_input(store, frame_id, ref):
    if (not isinstance(ref, dict) or set(ref) - {'frame_id', 'revision', 'input_hash'}
            or not isinstance(ref.get('revision'), int) or isinstance(ref['revision'], bool)
            or ref['revision'] < 1 or not isinstance(ref.get('input_hash'), str)):
        raise RecordedInputError('An exact input revision and input_hash are required for adaptation.')
    if ref.get('frame_id', frame_id) != frame_id:
        raise RecordedInputError('Adaptation input reference belongs to a different frame.')
    contract = next((item for item in store.versions('input', frame_id)
                     if item['revision'] == ref['revision']), None)
    if not contract or contract.get('input_hash') != ref['input_hash']:
        raise RecordedInputError('Adaptation input revision/hash does not match its frozen source.')
    if contract.get('protocol_version') not in STRUCTURED_PROTOCOLS:
        raise RecordedInputError('Adaptation requires a structured input with verified original views.')
    verify_input_images(store, contract)
    from .providers import validate_logged_request
    messages, _ = validate_logged_request(contract)
    if (len(messages) != 2 or messages[0] != {'role': 'system', 'content': contract['system_prompt']}
            or messages[1]['role'] != 'user' or not isinstance(messages[1]['content'], list)
            or [part['text'] for part in messages[1]['content'] if part['type'] == 'text'] != [contract['prompt']]):
        raise RecordedInputError('Adaptation source messages do not match their frozen prompt metadata.')
    return contract


def adapted_input_contract(store, frame_id, payload, canonical_image):
    """Build a six-section input from pinned parent and historical v1.0 template.

    The caller holds the Store transaction through this validation and append.
    Client text, image or provenance overrides are rejected, not trusted.
    """
    if any(key in payload for key in ('source_record_id', 'source_request', 'reconstruction_manifest_id', 'reconstruction', 'adaptation')):
        raise RecordedInputError('Adapted inputs cannot claim client-supplied archived or reconstructed provenance.')
    source = _pinned_input(store, frame_id, payload.get('adaptation_source'))
    current = store.latest('input', frame_id)
    if (source['revision'] != current['revision'] or source['input_hash'] != current['input_hash']
            or payload.get('expected_revision') != current['revision']):
        raise RecordedInputError('Adaptation must use the current input revision; reload before saving.')
    template_ref = payload.get('system_prompt_template')
    if not isinstance(template_ref, dict) or not isinstance(template_ref.get('frame_id'), str):
        raise RecordedInputError('An exact system_prompt_template frame, revision and input_hash are required.')
    template_frame = store.frame(template_ref['frame_id'])
    template = _pinned_input(store, template_ref['frame_id'], template_ref)
    if ('v1.0' not in template_frame.get('tags', []) or template.get('protocol_version') != PROTOCOL
            or expected_output_fields(template) != SIX_FIELDS or template.get('output_fields') != list(SIX_FIELDS)):
        raise RecordedInputError('The instruction template must be an archived v1.0 six-section input.')
    if canonical_image['sha256'] != source['image_sha256']:
        raise RecordedInputError('Canonical image changed since the source input was frozen.')
    keys = ('image_sha256', 'prompt', 'system_prompt', 'image_detail', 'velocity', 'track',
            'output_fields', 'image_layout', 'velocity_embedded_in_prompt', 'request_messages', 'input_images')
    contract = {key: copy.deepcopy(source[key]) for key in keys}
    contract.update(protocol_version=ADAPTED_PROTOCOL, system_prompt=template['system_prompt'],
                    output_fields=copy.deepcopy(template['output_fields']))
    contract['request_messages'][0] = {'role': 'system', 'content': template['system_prompt']}
    parent = {key: source[key] for key in ('frame_id', 'revision', 'input_hash', 'protocol_version')}
    parent.update(system_prompt_sha256=_sha(source['system_prompt']), output_fields=list(expected_output_fields(source)))
    for key in ('source_request', 'reconstruction', 'adaptation'):
        if key in source:
            parent[key] = copy.deepcopy(source[key])
    contract['adaptation'] = {
        'method_id': 'system_prompt_template.v1', 'historical_request_recovered': False,
        'parent_input': parent,
        'system_prompt_template': {**{key: template[key] for key in ('frame_id', 'revision', 'input_hash', 'protocol_version')},
                                   'system_prompt_sha256': _sha(template['system_prompt'])},
        'changed_fields': ['system_prompt', 'request_messages[0].content', 'output_fields'],
        'preserved_fields': ['prompt', 'request_messages[1]', 'velocity', 'input_images', 'image_sha256',
                             'image_detail', 'image_layout', 'velocity_embedded_in_prompt', 'track'],
        'note': 'New benchmark request derived from frozen evidence; historical outputs were not generated under this adapted request.'}
    for key in (*keys, 'protocol_version'):
        if key in payload and payload[key] != contract[key]:
            raise RecordedInputError('Edited ' + key + ' differs from the verified adaptation.')
    return contract


def archived_output_fields(contract):
    """Format of source archive evidence, separate from newly adapted outputs."""
    parent = contract or {}
    while parent.get('protocol_version') == ADAPTED_PROTOCOL:
        parent = parent.get('adaptation', {}).get('parent_input', {})
    return expected_output_fields(parent)
