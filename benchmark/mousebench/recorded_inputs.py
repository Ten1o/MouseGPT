"""Recover original per-frame requests without rewriting their text or images.

An archived request is evidence of its own contents, not proof that every
historical prediction was produced with that request. Reads fail closed when
source bytes, frame membership, or multiple archived requests disagree.
"""
import base64
import binascii
import copy
import hashlib
import json
import math
import re
from pathlib import Path


PROTOCOL = 'mousebench.input.v3.logged'
RECONSTRUCTED_PROTOCOL = 'mousebench.input.v3.reconstructed'
ADAPTED_PROTOCOL = 'mousebench.input.v3.adapted'
STRUCTURED_PROTOCOLS = (PROTOCOL, RECONSTRUCTED_PROTOCOL, ADAPTED_PROTOCOL)
_BASE_KEYS = ('protocol_version', 'image_sha256', 'prompt', 'system_prompt',
              'image_detail', 'velocity', 'track', 'output_fields', 'image_layout')
_IMAGE_KEYS = ('attachment_id', 'sha256', 'size', 'media_type', 'image_detail')
_SPEED_NAMES = ('overall', 'head', 'left_forelimb', 'right_forelimb',
                'left_hind_limb', 'right_hind_limb')
_NUMBER = r'\d+(?:\.\d+)?'
_SPEEDS = re.compile(
    rf'The overall movement speed of the mouse is:\s*(?P<overall>{_NUMBER})\s*cm/s,\s*'
    rf'the speed of the head:\s*(?P<head>{_NUMBER})\s*cm/s,\s*'
    r'speed of (?P<fore_first_side>left|right) and (?P<fore_second_side>left|right) forelimbs:\s*'
    rf'(?P<fore_first_value>{_NUMBER}),\s*(?P<fore_second_value>{_NUMBER})\s*cm/s,\s*'
    r'and (?P<hind_first_side>left|right) and (?P<hind_second_side>left|right) hind limbs are\s*'
    rf'(?P<hind_first_value>{_NUMBER}),\s*(?P<hind_second_value>{_NUMBER})\s*cm/s respectively\.')


class RecordedInputError(ValueError):
    def __init__(self, message, status='error'):
        super().__init__(message)
        self.status = status


def input_semantics(contract):
    """Return exactly the semantic fields hashed by Store.save_input.

    v2 contracts retain their established hash behavior. In v3, storage paths
    and provenance are not part of the request identity; ordered image bytes,
    image detail and exact message text are.
    """
    result = {key: copy.deepcopy(contract[key]) for key in _BASE_KEYS if key in contract}
    if contract.get('protocol_version') in STRUCTURED_PROTOCOLS:
        if isinstance(result.get('velocity'), dict):
            result['velocity'].pop('source', None)
        result['velocity_embedded_in_prompt'] = contract['velocity_embedded_in_prompt']
        result['request_messages'] = copy.deepcopy(contract['request_messages'])
        result['input_images'] = [{key: image[key] for key in _IMAGE_KEYS}
                                  for image in contract['input_images']]
    return result


def verify_input_images(store, contract):
    """Verify the frozen recorded request and every original view before use.

    Manual v2 contracts continue to use Store.image and return an empty list.
    """
    if contract.get('protocol_version') not in STRUCTURED_PROTOCOLS:
        return []
    images = contract.get('input_images')
    if not isinstance(images, list) or len(images) != 2:
        raise RecordedInputError('Recorded input requires exactly two original view images.')
    try:
        semantic_json = json.dumps(input_semantics(contract), ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False)
    except (KeyError, TypeError, ValueError) as exc:
        raise RecordedInputError('Recorded input contract is incomplete or malformed.') from exc
    if hashlib.sha256(semantic_json.encode()).hexdigest() != contract.get('input_hash'):
        raise RecordedInputError('Recorded input contract hash mismatch; restore its original frozen request.')
    for index, image in enumerate(images, 1):
        if (not isinstance(image, dict) or image.get('attachment_id') != 'input_image_' + str(index)
                or image.get('media_type') != 'image/jpeg' or image.get('image_detail') not in ('high', 'low', 'auto')
                or not isinstance(image.get('size'), int) or isinstance(image['size'], bool) or image['size'] <= 0):
            raise RecordedInputError('Recorded input image metadata is invalid.')
        relative = image.get('path')
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise RecordedInputError('Recorded input image path must be inside the benchmark data directory.')
        path = (store.root / relative).resolve()
        if not path.is_relative_to(store.root) or not path.is_file():
            raise RecordedInputError('Recorded input image is missing or outside the benchmark data directory.')
        data = path.read_bytes()
        if len(data) != image['size'] or hashlib.sha256(data).hexdigest() != image.get('sha256'):
            raise RecordedInputError('Recorded input image checksum or size mismatch; restore the original source bytes.')
    return copy.deepcopy(images)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecordedInputError('Archived request contains a duplicate JSON key: ' + key)
        result[key] = value
    return result


def _read_source(store, frame_id, record):
    if record.get('frame_id') != frame_id or record.get('frame_membership_verified') is not True:
        raise RecordedInputError('Archived request frame membership is not verified for this frame.')
    relative = record.get('path')
    if not isinstance(relative, str) or not relative:
        raise RecordedInputError('Archived request has no local source path.')
    path = (store.root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(store.root) or not path.is_file():
        raise RecordedInputError('Archived request source must be an existing file inside the benchmark data directory.')
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != record.get('sha256'):
        raise RecordedInputError('Archived request checksum mismatch; restore its original source bytes.')
    try:
        request = json.loads(raw.decode('utf-8'), object_pairs_hook=_no_duplicate_keys,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite JSON value')))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RecordedInputError('Archived request is not valid JSON: ' + str(exc)) from exc
    if not isinstance(request, dict):
        raise RecordedInputError('Archived request must be a JSON object.')
    return request, sha


def _parse(store, frame_id, record):
    request, sha = _read_source(store, frame_id, record)
    return _parse_request(request, sha, record)


def _parse_request(request, sha, record):
    """Shared exact request validation; caller supplies trustworthy provenance."""
    messages = request.get('messages')
    if (not isinstance(messages, list) or len(messages) != 2
            or not all(isinstance(message, dict) for message in messages)
            or messages[0].get('role') != 'system' or messages[1].get('role') != 'user'
            or set(messages[0]) != {'role', 'content'} or set(messages[1]) != {'role', 'content'}):
        raise RecordedInputError('Expected the original system and user input messages only; unsupported messages require review.')
    system = messages[0]['content']
    parts = messages[1]['content']
    if not isinstance(system, str) or not system.strip() or not isinstance(parts, list):
        raise RecordedInputError('Archived system text or user content is missing or malformed.')
    text_parts, normalized, images, image_bytes = [], [], [], []
    for part in parts:
        if not isinstance(part, dict):
            raise RecordedInputError('Archived user content contains an invalid part.')
        if part.get('type') == 'text':
            if set(part) != {'type', 'text'} or not isinstance(part.get('text'), str):
                raise RecordedInputError('Archived user text is malformed.')
            text_parts.append(part['text'])
            normalized.append(copy.deepcopy(part))
        elif part.get('type') == 'image_url':
            image = part.get('image_url')
            if (set(part) != {'type', 'image_url'} or not isinstance(image, dict)
                    or set(image) != {'url', 'detail'} or image.get('detail') not in ('high', 'low', 'auto')):
                raise RecordedInputError('Archived image reference or detail is malformed.')
            url = image.get('url')
            prefix = 'data:image/jpeg;base64,'
            if not isinstance(url, str) or not url.startswith(prefix):
                raise RecordedInputError('Archived request requires embedded original JPEG bytes; remote or other image types need review.')
            try:
                data = base64.b64decode(url[len(prefix):], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise RecordedInputError('Archived image contains invalid base64 data.') from exc
            if not data.startswith(b'\xff\xd8\xff') or not data.endswith(b'\xff\xd9'):
                raise RecordedInputError('Archived image bytes are not a complete JPEG.')
            attachment_id = 'input_image_' + str(len(images) + 1)
            image_sha = hashlib.sha256(data).hexdigest()
            images.append({'attachment_id': attachment_id, 'path': 'objects/' + image_sha + '.jpg',
                           'sha256': image_sha, 'size': len(data), 'media_type': 'image/jpeg',
                           'image_detail': image['detail']})
            image_bytes.append(data)
            normalized.append({'type': 'image_reference', 'attachment_id': attachment_id})
        else:
            raise RecordedInputError('Archived user content has an unsupported part; it must not be silently dropped.')
    if len(text_parts) != 1 or not text_parts[0].strip() or len(images) != 2:
        raise RecordedInputError('Expected one exact per-frame prompt and two original view images; source requires review.')
    prompt = text_parts[0]
    matches = list(_SPEEDS.finditer(prompt))
    if len(matches) != 1:
        raise RecordedInputError('Could not identify exactly one set of six recorded speeds; source requires review.')
    values = matches[0].groupdict()
    speeds = {name: float(values[name]) for name in ('overall', 'head')}
    # Older request templates name right then left, while the v1.0 source
    # names left then right. Map each pair by its own explicit side labels;
    # never reorder or rewrite the archived prompt itself.
    for prefix, limb in (('fore', 'forelimb'), ('hind', 'hind_limb')):
        sides = [values[prefix + '_' + position + '_side'] for position in ('first', 'second')]
        if set(sides) != {'left', 'right'}:
            raise RecordedInputError('Recorded limb speeds require one explicit left and one right side; source requires review.')
        for position, side in zip(('first', 'second'), sides):
            speeds[side + '_' + limb] = float(values[prefix + '_' + position + '_value'])
    speeds = {name: speeds[name] for name in _SPEED_NAMES}
    if not all(math.isfinite(value) and value >= 0 for value in speeds.values()):
        raise RecordedInputError('Recorded speed values must be finite and non-negative.')
    settings = {key: request[key] for key in ('top_p', 'temperature', 'max_tokens', 'seed') if key in request}
    for key, value in settings.items():
        if key == 'seed':
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise RecordedInputError('Archived generation setting is invalid: seed')
            continue
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
            raise RecordedInputError('Archived generation setting is invalid: ' + key)
    source_path = record.get('source_path') or record.get('nas_path')
    if not isinstance(source_path, str) or not source_path:
        raise RecordedInputError('Archived request has no recorded source provenance.')
    preview = {'status': 'ready', 'source_record_id': record['id'], 'source_path': source_path,
               'source_sha256': sha, 'system_prompt': system, 'prompt': prompt,
               'velocity': {'value': speeds, 'units': 'cm/s', 'source': source_path},
               'velocity_embedded_in_prompt': True, 'track': 'image_velocity',
               'image_layout': 'historical_separate_views', 'image_detail': images[0]['image_detail'],
               'input_images': images, 'generation_settings': settings,
               'request_messages': [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': normalized}]}
    return preview, image_bytes


def resolve_recorded_input(store, frame_id, source_record_id=None, materialize=False):
    """Verify all archived inference requests and resolve a single exact input."""
    store.frame(frame_id)
    records = [record for record in store.get_records('raw_records', frame_id)
               if record.get('role') == 'inference_log']
    if not records:
        raise RecordedInputError('No archived inference request is available for this frame.', 'unavailable')
    if source_record_id is not None and source_record_id not in {record['id'] for record in records}:
        raise RecordedInputError('The selected request does not belong to this frame.')
    parsed = [_parse(store, frame_id, record) for record in records]
    # Copies may differ in source provenance; input text, image order/bytes and
    # recorded settings must agree. A selected ID is not a conflict waiver.
    identities = []
    for preview, _ in parsed:
        identities.append(json.dumps({'request_messages': preview['request_messages'],
                                      'input_images': [{key: image[key] for key in _IMAGE_KEYS} for image in preview['input_images']],
                                      'generation_settings': preview['generation_settings']},
                                     ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False))
    if len(set(identities)) != 1:
        raise RecordedInputError('Conflicting archived requests exist for this frame; resolve the source conflict before saving an input.', 'conflict')
    preview, blobs = next(((value, blobs) for value, blobs in parsed
                           if source_record_id is None or value['source_record_id'] == source_record_id))
    if materialize:
        objects = store.root / 'objects'
        objects.mkdir(exist_ok=True)
        if not objects.resolve().is_relative_to(store.root):
            raise RecordedInputError('Input object directory must remain inside the benchmark data directory.')
        for image, blob in zip(preview['input_images'], blobs):
            path = store.root / image['path']
            if not path.resolve().is_relative_to(store.root):
                raise RecordedInputError('Input image object must remain inside the benchmark data directory.')
            try:
                with path.open('xb') as handle:
                    handle.write(blob)
            except FileExistsError:
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != image['sha256']:
                    raise RecordedInputError('Existing input image object checksum mismatch; original bytes were not overwritten.')
    return preview


def recorded_input_preview(store, frame_id):
    """Read-only UI summary; errors are explicit and never become default prompts."""
    try:
        return resolve_recorded_input(store, frame_id)
    except (OSError, ValueError) as exc:
        return {'status': getattr(exc, 'status', 'error'), 'error': str(exc)}


def recorded_input(store, frame_id):
    """Public read-only preview, shared by UI and batch preparation."""
    return recorded_input_preview(store, frame_id)


def recorded_input_contract(store, frame_id, payload, canonical_image, output_fields):
    """Resolve a save request from its source ID, never from client-edited text."""
    if (store.latest('input', frame_id) or {}).get('protocol_version') == ADAPTED_PROTOCOL:
        raise RecordedInputError('Archived import cannot overwrite a standardized input; its original request remains in input history.')
    source_id = payload.get('source_record_id')
    if not isinstance(source_id, str) or not source_id.strip():
        raise RecordedInputError('A recorded source request ID is required.')
    recorded = resolve_recorded_input(store, frame_id, source_id)
    for key in ('prompt', 'system_prompt', 'velocity', 'image_detail', 'input_images',
                'request_messages', 'velocity_embedded_in_prompt', 'image_layout', 'track'):
        if key in payload and payload[key] != recorded[key]:
            raise RecordedInputError('The edited ' + key + ' does not match the archived request. Use the unchanged source input.')
    # Validate before materializing; immutable files are kept by content hash.
    recorded = resolve_recorded_input(store, frame_id, source_id, materialize=True)
    contract = {key: recorded[key] for key in ('prompt', 'system_prompt', 'velocity', 'image_detail',
                'input_images', 'request_messages', 'velocity_embedded_in_prompt', 'image_layout', 'track')}
    contract.update(protocol_version=PROTOCOL, image_sha256=canonical_image['sha256'],
                    output_fields=list(output_fields),
                    source_request={'record_id': recorded['source_record_id'], 'source_path': recorded['source_path'],
                                    'sha256': recorded['source_sha256'], 'generation_settings': recorded['generation_settings']})
    return contract
