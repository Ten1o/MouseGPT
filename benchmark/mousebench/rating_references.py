"""Read-only rating/output references; matching a label never verifies a rating.

Case-insensitive equality and two explicit browsing aliases supply candidates.
No prefix matching, source repair, or model-identity inference is used.
Human links are checked against current source rows and ordered exact outputs.
"""
import hashlib
import json

from .evaluation import analyze_output, OUTPUT_FIELDS


# Browsing conveniences observed in this archive, never model-identity claims.
# Keep checkpoint/version suffixes intact; in particular Turbo is not GPT-4o.
_BROWSING_ALIASES = (frozenset(('minicpm-2.6', 'minicpm-v-2_6')),
                     frozenset(('gpt4o', 'gpt-4o-2024-05-13')))


def _browsing_names(label):
    normalized = label.casefold()
    return next((group for group in _BROWSING_ALIASES if normalized in group), {normalized})


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _labels(output):
    labels = {output['model_label']} if isinstance(output.get('model_label'), str) and output['model_label'] else set()
    for key in ('path_model_labels', 'observed_model_directories'):
        if isinstance(output.get(key), list):
            labels.update(value for value in output[key] if isinstance(value, str) and value)
    return labels


def _prediction_text(prediction):
    text = prediction.get('output_text', prediction.get('output'))
    if not isinstance(text, str):
        return None
    if prediction.get('output_text') is not None and prediction.get('output') is not None and prediction['output_text'] != prediction['output']:
        return None
    return text


def _checked_link(link, rating, frame_id, prediction_lookup, side_count):
    """Return ordered current predictions, or an explicit rejection reason."""
    if link.get('frame_id') != frame_id:
        return None, 'the link belongs to a different frame'
    if link.get('human_confirmed') is not True:
        return None, 'human confirmation is missing'
    hashes = link.get('source_rating_hashes')
    if not isinstance(hashes, dict) or hashes.get(rating['id']) != _digest(rating):
        return None, 'the source rating has changed or its saved hash is missing'
    run_ids = [link.get('run_id')]
    if side_count == 2:
        run_ids.append(link.get('other_run_id'))
    elif link.get('other_run_id'):
        return None, 'a single-model rating has a two-output link'
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids) or len(set(run_ids)) != side_count:
        return None, 'the ordered run identifiers are missing or invalid'
    linked_hashes = link.get('prediction_hashes')
    if not isinstance(linked_hashes, list) or len(linked_hashes) != side_count:
        return None, 'the ordered output hashes are missing or invalid'
    predictions = []
    for run_id, expected in zip(run_ids, linked_hashes):
        matches = prediction_lookup.get((run_id, frame_id), [])
        if len(matches) != 1 or matches[0].get('status') != 'succeeded':
            return None, 'a linked run no longer has one successful output for this frame'
        prediction = matches[0]
        text = _prediction_text(prediction)
        if (text is None or not isinstance(expected, str)
            or prediction.get('prediction_sha256') != expected
            or hashlib.sha256(text.encode()).hexdigest() != expected):
            return None, 'a linked output or its ordered hash does not match the current prediction'
        predictions.append(prediction)
    return predictions, None


def resolve_rating_references(frame_id, ratings, descriptions, evidence_links=(), predictions=(), runs=(), expected_fields=OUTPUT_FIELDS):
    """Return display-only references and any additional human-linked outputs.

    Model/path/directory labels, case variants and explicit browsing aliases
    supply candidates only; checkpoint suffixes are never collapsed.
    All copies remain visible; differing texts are ambiguous. A verified status
    binds the rated output text, not the declared model identity or scientific
    correctness. A/B always follows the rating and human link's stored order.
    """
    descriptions = [record for record in descriptions if record.get('frame_id') == frame_id
                    and isinstance(record.get('id'), str) and isinstance(record.get('text'), str)]
    prediction_lookup = {}
    for prediction in predictions:
        prediction_lookup.setdefault((prediction.get('run_id'), prediction.get('frame_id')), []).append(prediction)
    run_lookup = {run['id']: run for run in runs if isinstance(run, dict) and isinstance(run.get('id'), str)}
    label_lookup = {}
    for output in descriptions:
        for label in _labels(output):
            label_lookup.setdefault(label.casefold(), []).append(output)
    extras = {}

    def prediction_ids(prediction):
        text = _prediction_text(prediction)
        # Identical output text does not establish identical archive provenance
        # (different models can return the same empty/error/template response).
        asset = prediction.get('source_asset_id')
        direct = [output['id'] for output in descriptions if asset and output['id'] == asset and output['text'] == text]
        if direct:
            return sorted(direct)
        source = prediction.get('source_path') or prediction.get('source_locator')
        located = [output['id'] for output in descriptions if isinstance(source, str) and source
                   and output['text'] == text and source in
                   (output.get('source_path'), output.get('nas_path'), output.get('record_locator'))]
        if located:
            return sorted(located)
        output_id = 'rating_prediction_' + _digest([prediction['run_id'], frame_id, prediction['prediction_sha256']])[:24]
        if output_id not in extras:
            metadata = prediction.get('model_metadata') or {}
            extras[output_id] = {'id': output_id, 'frame_id': frame_id, 'text': text,
                'model_label': run_lookup.get(prediction['run_id'], {}).get('model') or metadata.get('directory_label') or metadata.get('model_id') or '',
                'run_id': prediction['run_id'], 'prediction_sha256': prediction['prediction_sha256'],
                'output_view': analyze_output(text, prediction.get('expected_output_fields', expected_fields))}
            if source:
                extras[output_id]['source_path'] = source
        return [output_id]

    references = {}
    for rating in ratings:
        if rating.get('frame_id') != frame_id or not isinstance(rating.get('id'), str):
            continue
        pairwise = rating.get('kind') == 'pairwise'
        specs = [('A', 'model_a', 'key_model_a'), ('B', 'model_b', 'key_model_b')] if pairwise else [('single', 'model_label', 'path_model_label')]
        warnings, valid_links = [], []
        for link in evidence_links:
            if not isinstance(link, dict) or rating['id'] not in (link.get('rating_ids') or []):
                continue
            checked, error = _checked_link(link, rating, frame_id, prediction_lookup, len(specs))
            if error:
                warnings.append('Saved link ' + str(link.get('id', '(unnamed)')) + ' is unverified: ' + error + '.')
            else:
                valid_links.append(checked)
        conflicting_links = len({tuple(prediction['prediction_sha256'] for prediction in link) for link in valid_links}) > 1
        if conflicting_links:
            warnings.append('Human-confirmed links disagree about the ordered rated outputs; no link is selected as authoritative.')
        sides = []
        for index, (side, primary_key, secondary_key) in enumerate(specs):
            primary, secondary = rating.get(primary_key), rating.get(secondary_key)
            primary = primary if isinstance(primary, str) else ''
            secondary = secondary if isinstance(secondary, str) else ''
            labels = {label for label in (primary, secondary) if label}
            source_conflict = bool(primary and secondary and primary != secondary)
            browsing_names = {name for label in labels for name in _browsing_names(label)}
            candidates = {output['id']: output for name in browsing_names for output in label_lookup.get(name, [])}
            uses_alias = any(not labels.intersection(_labels(output)) for output in candidates.values())
            linked = sorted({output_id for link in valid_links for output_id in prediction_ids(link[index])})
            if source_conflict:
                warnings.append(side + ': model-cell label ' + repr(primary) + ' disagrees with the path/key label ' + repr(secondary) + '; both candidate sets are retained.')
            if source_conflict or conflicting_links:
                status = 'conflict'
                output_ids = sorted(set(candidates) | set(linked))
                reason = 'Source labels or verified links conflict. These references do not establish which answer was rated.'
            elif valid_links:
                status, output_ids = 'verified', linked
                reason = 'A human-confirmed link matches the current source-rating hash and ordered successful output hashes. This verifies output correspondence, not model identity or correctness.'
            else:
                output_ids = sorted(candidates)
                if not output_ids:
                    status = 'unavailable'
                    reason = 'No saved output has a same-frame label or explicit browsing-alias match; no fuzzy or prefix match was substituted.'
                elif len({output['text'] for output in candidates.values()}) > 1:
                    status = 'ambiguous'
                    reason = 'Same-frame label candidates match differing saved answers. All copies are shown; the rated answer is unverified.'
                else:
                    status = 'candidate'
                    reason = 'Same-frame label candidate only; correspondence to the rated answer is unverified.'
                    if len(output_ids) > 1:
                        reason += ' Multiple copies have identical text.'
            if uses_alias and status != 'verified':
                alias_note = 'Case/name aliases are browsing candidates only, not verification of exact model or export identity.'
                warnings.append(side + ': ' + alias_note)
                reason += ' ' + alias_note
            for output in candidates.values():
                metadata_labels = {value for value in output.get('actual_models', []) if isinstance(value, str) and value} if isinstance(output.get('actual_models'), list) else set()
                metadata_names = {value.casefold() for value in metadata_labels}
                model_label = output.get('model_label')
                if (isinstance(model_label, str) and model_label and model_label.casefold() not in browsing_names) or metadata_names - browsing_names:
                    warnings.append(side + ': matching path/directory labels differ from saved model metadata. Original labels are preserved; the model identity is not established by this reference.')
            sides.append({'side': side, 'model_label': primary or secondary, 'status': status,
                          'reason': reason, 'output_ids': output_ids})
        references[rating['id']] = {'kind': 'pairwise' if pairwise else 'single',
                                   'warnings': list(dict.fromkeys(warnings)), 'sides': sides}
    return {'rating_output_references': references,
            'rating_reference_outputs': [extras[key] for key in sorted(extras)]}
