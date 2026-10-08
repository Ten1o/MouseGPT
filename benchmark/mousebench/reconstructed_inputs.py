"""Explicitly reconstructed requests, kept separate from archived execution logs."""
import copy
import hashlib
import json

from .recorded_inputs import (PROTOCOL, RECONSTRUCTED_PROTOCOL, _SPEEDS, _read_source,
                              _parse_request, verify_input_images, RecordedInputError)


def speed_template(prompt):
    """Mask only the six numeric speed values, preserving every other character."""
    matches = list(_SPEEDS.finditer(prompt))
    if len(matches) != 1:
        raise RecordedInputError('Reconstruction must retain one explicit six-speed prompt clause.')
    match = matches[0]
    spans = []
    for name in ('overall', 'head'):
        spans.append((*match.span(name), '{{'+name+'}}'))
    for prefix, limb in (('fore', 'forelimb'), ('hind', 'hind_limb')):
        for position in ('first', 'second'):
            side = match.group(prefix+'_'+position+'_side')
            spans.append((*match.span(prefix+'_'+position+'_value'), '{{'+side+'_'+limb+'}}'))
    result = prompt
    for start, end, replacement in sorted(spans, reverse=True):
        result = result[:start]+replacement+result[end:]
    return result


def reconstructed_input_contract(store, frame_id, payload, canonical_image):
    """Read a verified reconstruction_manifest asset and freeze its exact request.

    The manifest is not an inference_log, and this path never creates one.
    Historical input versions cannot be replaced through this importer.
    """
    latest = store.latest('input', frame_id)
    if latest and latest.get('protocol_version') != RECONSTRUCTED_PROTOCOL:
        raise RecordedInputError('A reconstruction cannot replace an existing historical or manual input.')
    source_id = payload.get('reconstruction_manifest_id')
    record = next((row for row in store.get_records('assets', frame_id)
                   if row['id'] == source_id and row.get('role') == 'reconstruction_manifest'), None)
    if not record:
        raise RecordedInputError('A reconstruction_manifest asset for this frame is required.')
    manifest, sha = _read_source(store, frame_id, record)
    if manifest.get('frame_id') != frame_id or not isinstance(manifest.get('request'), dict):
        raise RecordedInputError('Reconstruction manifest frame/request is invalid.')
    provenance = manifest.get('reconstruction')
    if not isinstance(provenance, dict):
        raise RecordedInputError('Explicit reconstruction provenance is required.')
    if not isinstance(provenance.get('method_id'), str) or not provenance['method_id'].strip():
        raise RecordedInputError('A reconstruction method_id is required.')
    for key in ('velocity_method', 'image_method'):
        if not isinstance(provenance.get(key), (dict, str)) or not provenance[key]:
            raise RecordedInputError('Reconstruction requires documented '+key+'.')
    limitations = provenance.get('limitations')
    if not isinstance(limitations, list) or any(not isinstance(value, str) for value in limitations):
        raise RecordedInputError('Reconstruction limitations must be a list of text notes.')
    template_ids = provenance.get('template_frame_ids')
    if (not isinstance(template_ids, list) or not template_ids or any(not isinstance(fid, str) for fid in template_ids)
            or len(template_ids) != len(set(template_ids))):
        raise RecordedInputError('Distinct source template frame IDs are required.')
    preview, blobs = _parse_request(manifest['request'], sha, record)
    template_hashes = {}
    for fid in template_ids:
        template = store.latest('input', fid)
        if not template or template.get('protocol_version') != PROTOCOL:
            raise RecordedInputError('Template frames must have actual archived input contracts.')
        verify_input_images(store, template)
        if (template['system_prompt'] != preview['system_prompt']
                or speed_template(template['prompt']) != speed_template(preview['prompt'])):
            raise RecordedInputError('Reconstructed text differs from its template beyond six speed values.')
        if [item['image_detail'] for item in template['input_images']] != [item['image_detail'] for item in preview['input_images']]:
            raise RecordedInputError('Reconstructed image detail/order does not match the template specification.')
        template_hashes[fid] = template['input_hash']
    from .output_format import expected_output_fields
    fields = expected_output_fields(preview)
    if 'output_fields' in manifest and manifest['output_fields'] != list(fields):
        raise RecordedInputError('Declared output fields differ from the original template requirement.')
    if 'source_request' in payload or 'source_record_id' in payload:
        raise RecordedInputError('Reconstructed inputs cannot claim archived request provenance.')
    contract = {key: copy.deepcopy(preview[key]) for key in ('prompt', 'system_prompt', 'velocity', 'image_detail',
                'input_images', 'request_messages', 'velocity_embedded_in_prompt', 'track')}
    contract.update(protocol_version=RECONSTRUCTED_PROTOCOL, image_sha256=canonical_image['sha256'],
                    output_fields=list(fields), image_layout='reconstructed_separate_views')
    contract['reconstruction'] = dict(copy.deepcopy(provenance),
        manifest_id=record['id'], manifest_sha256=sha, manifest_path=record['path'],
        manifest_source_path=record.get('source_path') or record.get('nas_path'),
        template_input_hashes=template_hashes, generation_settings=preview['generation_settings'],
        historical_request_recovered=False)
    # Client-supplied content cannot silently override the immutable manifest.
    for key in ('prompt','system_prompt','velocity','image_detail','input_images','request_messages','output_fields'):
        if key in payload and payload[key] != contract[key]:
            raise RecordedInputError('Edited '+key+' differs from the reconstruction manifest.')
    objects = store.root/'objects'
    objects.mkdir(exist_ok=True)
    if not objects.resolve().is_relative_to(store.root):
        raise RecordedInputError('Reconstructed image storage must remain inside the data directory.')
    for image, raw in zip(contract['input_images'], blobs):
        path = store.root/image['path']
        if not path.resolve().is_relative_to(store.root):
            raise RecordedInputError('Reconstructed image storage must remain inside the data directory.')
        if path.exists():
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != image['sha256']:
                raise RecordedInputError('Existing reconstructed image checksum mismatch.')
        else:
            with path.open('xb') as handle:
                handle.write(raw)
    return contract
