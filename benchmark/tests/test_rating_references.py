"""Display linkage tests: explicit browsing aliases never establish identity."""
import copy
import hashlib
import json
import tempfile
import unittest

from mousebench.rating_references import resolve_rating_references
from mousebench.store import Store, digest
from mousebench.workflow import Workflow

FRAME = '240420_C10M1_00003972'
OTHER = '240420_C10M1_00003973'


def rating(**changes):
    return dict({'id': 'rating-1', 'frame_id': FRAME, 'kind': 'rubric',
                 'model_label': 'model-a', 'path_model_label': 'model-a',
                 'value': {'Head': 1, 'Limb': 0}}, **changes)


def pair(**changes):
    return dict({'id': 'rating-pair', 'frame_id': FRAME, 'kind': 'pairwise',
                 'model_a': 'model-a', 'key_model_a': 'model-a',
                 'model_b': 'model-b', 'key_model_b': 'model-b', 'value': 'B is better'}, **changes)


def output(identity='out-a', model='model-a', text='Fixture answer.', **changes):
    return dict({'id': identity, 'frame_id': FRAME, 'model_label': model, 'text': text}, **changes)


def prediction(run='run-a', text='Fixture linked answer.', **changes):
    return dict({'run_id': run, 'frame_id': FRAME, 'status': 'succeeded', 'output_text': text,
                 'prediction_sha256': hashlib.sha256(text.encode()).hexdigest()}, **changes)


def link(source, predictions, **changes):
    return dict({'id': 'link-1', 'frame_id': FRAME, 'rating_ids': [source['id']],
                 'source_rating_hashes': {source['id']: digest(source)}, 'human_confirmed': True,
                 'run_id': predictions[0]['run_id'],
                 'other_run_id': predictions[1]['run_id'] if len(predictions) > 1 else None,
                 'prediction_hashes': [p['prediction_sha256'] for p in predictions]}, **changes)


class RatingReferenceTests(unittest.TestCase):
    def resolve(self, source=None, descriptions=(), links=(), predictions=(), runs=()):
        source = rating() if source is None else source
        result = resolve_rating_references(FRAME, [source], descriptions, links, predictions, runs)
        return result['rating_output_references'][source['id']], result['rating_reference_outputs']

    def test_single_section_rating_unique_match_is_candidate_never_verified(self):
        source = rating()
        reference, extras = self.resolve(source, [output()])
        self.assertEqual(reference['kind'], 'single')
        self.assertEqual(reference['sides'][0]['side'], 'single')
        self.assertEqual(reference['sides'][0]['status'], 'candidate')
        self.assertEqual(reference['sides'][0]['output_ids'], ['out-a'])
        self.assertEqual(source['value'], {'Head': 1, 'Limb': 0})
        self.assertEqual(extras, [])

    def test_pairwise_preserves_ab_order_regardless_of_answer_sort_or_winner(self):
        reference, _ = self.resolve(pair(), [output('first-sorted', 'model-b'), output('last-sorted')])
        self.assertEqual([side['side'] for side in reference['sides']], ['A', 'B'])
        self.assertEqual([side['output_ids'] for side in reference['sides']], [['last-sorted'], ['first-sorted']])
        self.assertEqual([side['model_label'] for side in reference['sides']], ['model-a', 'model-b'])

    def test_distinct_copies_are_ambiguous_and_identical_copies_all_remain(self):
        copies = [output(), output('copy', text='Different fixture answer.')]
        reference, _ = self.resolve(descriptions=copies)
        self.assertEqual(reference['sides'][0]['status'], 'ambiguous')
        self.assertEqual(reference['sides'][0]['output_ids'], ['copy', 'out-a'])
        copies[1]['text'] = copies[0]['text']
        reference, _ = self.resolve(descriptions=copies)
        self.assertEqual(reference['sides'][0]['status'], 'candidate')
        self.assertEqual(len(reference['sides'][0]['output_ids']), 2)
        self.assertIn('identical text', reference['sides'][0]['reason'])

    def test_missing_prefix_collision_and_cross_frame_do_not_match(self):
        candidates = [output('prefix', 'model-a-finetuned'),
                      output('cross-frame', frame_id=OTHER), output('metadata-only', 'other', actual_models=['model-a'])]
        reference, _ = self.resolve(descriptions=candidates)
        self.assertEqual(reference['sides'][0]['status'], 'unavailable')
        self.assertEqual(reference['sides'][0]['output_ids'], [])
        result = resolve_rating_references(FRAME, [rating(frame_id=OTHER)], candidates)
        self.assertEqual(result['rating_output_references'], {})

    def test_case_variants_are_explicitly_unverified_browsing_candidates(self):
        source = rating(model_label='internvl2-llama3-76b', path_model_label='internvl2-llama3-76b')
        reference, _ = self.resolve(source, [output(model='InternVL2-Llama3-76B')])
        side = reference['sides'][0]
        self.assertEqual(side['status'], 'candidate')
        self.assertEqual(side['output_ids'], ['out-a'])
        self.assertIn('Case/name aliases', side['reason'])
        self.assertTrue(any('not verification' in warning for warning in reference['warnings']))

    def test_explicit_browsing_aliases_are_symmetric_without_collapsing_versions(self):
        for first, second in (('minicpm-2.6', 'MiniCPM-V-2_6'), ('gpt4o', 'gpt-4o-2024-05-13')):
            for rated, saved in ((first, second), (second, first)):
                with self.subTest(rated=rated, saved=saved):
                    source = rating(model_label=rated, path_model_label=rated)
                    reference, _ = self.resolve(source, [output(model=saved)])
                    self.assertEqual(reference['sides'][0]['status'], 'candidate')
                    self.assertIn('Case/name aliases', reference['sides'][0]['reason'])
                    self.assertEqual(reference['sides'][0]['model_label'], rated)
        for rated, saved in (('minicpm-2.6', 'minicpm-2.6-full-fp32-45900'),
                             ('minicpm-2.6', 'MiniCPM-V-2_6-lora-1200'),
                             ('gpt4o', 'gpt-4-turbo-2024-04-09'),
                             ('gpt4o', 'gpt-4o-2024-08-06'), ('gpt4o', 'gpt-4o-mini')):
            with self.subTest(rated=rated, saved=saved):
                source = rating(model_label=rated, path_model_label=rated)
                self.assertEqual(self.resolve(source, [output(model=saved)])[0]['sides'][0]['status'], 'unavailable')

    def test_exact_and_alias_outputs_are_combined_and_differences_stay_ambiguous(self):
        source = rating(model_label='minicpm-2.6', path_model_label='minicpm-2.6')
        descriptions = [output('exact', 'minicpm-2.6', 'Exact-label answer.'),
                        output('alias', 'MiniCPM-V-2_6', 'Alias-label different answer.')]
        reference, _ = self.resolve(source, descriptions)
        self.assertEqual(reference['sides'][0]['status'], 'ambiguous')
        self.assertEqual(reference['sides'][0]['output_ids'], ['alias', 'exact'])
        self.assertIn('Case/name aliases', reference['sides'][0]['reason'])

    def test_exact_path_and_directory_labels_work_without_renaming_turbo(self):
        description = output(model='gpt-4-turbo-2024-04-09', path_model_labels=['gpt4o'],
                             actual_models=['gpt-4-turbo-2024-04-09'])
        original = copy.deepcopy(description)
        reference, _ = self.resolve(rating(model_label='gpt4o', path_model_label='gpt4o'), [description])
        self.assertEqual(reference['sides'][0]['status'], 'candidate')
        self.assertEqual(reference['sides'][0]['model_label'], 'gpt4o')
        self.assertTrue(reference['warnings'])
        self.assertEqual(description, original)
        directory = output('directory', model='', observed_model_directories=['model-a'])
        self.assertEqual(self.resolve(descriptions=[directory])[0]['sides'][0]['output_ids'], ['directory'])

    def test_single_cell_path_conflict_retains_both_candidate_sets(self):
        source = rating(path_model_label='model-b')
        reference, _ = self.resolve(source, [output(), output('out-b', 'model-b')])
        self.assertEqual(reference['sides'][0]['status'], 'conflict')
        self.assertEqual(reference['sides'][0]['output_ids'], ['out-a', 'out-b'])
        self.assertIn('disagrees', reference['warnings'][0])

    def test_pair_key_conflict_affects_correct_side_without_swapping(self):
        reference, _ = self.resolve(pair(key_model_a='model-c'),
                                    [output(), output('out-b', 'model-b'), output('out-c', 'model-c')])
        self.assertEqual([side['status'] for side in reference['sides']], ['conflict', 'candidate'])
        self.assertEqual(reference['sides'][0]['output_ids'], ['out-a', 'out-c'])
        self.assertEqual(reference['sides'][1]['output_ids'], ['out-b'])

    def test_human_link_prefers_exact_source_asset_and_text(self):
        source, pred = rating(), prediction(text='Fixture answer.', source_asset_id='out-a')
        reference, extras = self.resolve(source, [output(), output('copy'), output('other', text='Different.')],
                                         [link(source, [pred])], [pred])
        self.assertEqual(reference['sides'][0]['status'], 'verified')
        self.assertEqual(reference['sides'][0]['output_ids'], ['out-a'])
        self.assertEqual(extras, [])

    def test_human_link_can_reuse_exact_source_locator_and_text(self):
        source, pred = rating(), prediction(text='Fixture answer.', source_locator='/fixture/a.txt')
        descriptions = [output(source_path='/fixture/a.txt'), output('copy', nas_path='/fixture/a.txt'),
                        output('unrelated-model', model='model-b', source_path='/fixture/b.txt')]
        reference, extras = self.resolve(source, descriptions, [link(source, [pred])], [pred])
        self.assertEqual(reference['sides'][0]['output_ids'], ['copy', 'out-a'])
        self.assertEqual(extras, [])

    def test_identical_text_alone_cannot_transfer_unrelated_archive_provenance(self):
        source, pred = rating(), prediction(text='Fixture answer.')
        descriptions = [output(), output('unrelated-model', model='model-b')]
        reference, extras = self.resolve(source, descriptions, [link(source, [pred])], [pred])
        self.assertEqual(reference['sides'][0]['status'], 'verified')
        self.assertEqual(reference['sides'][0]['output_ids'], [extras[0]['id']])
        self.assertNotIn('source_path', extras[0])
        # An asset ID with different current text likewise cannot be reused.
        pred['source_asset_id'] = 'out-a'
        descriptions[0]['text'] = 'Changed archive text.'
        reference, extras = self.resolve(source, descriptions, [link(source, [pred])], [pred])
        self.assertEqual(reference['sides'][0]['output_ids'], [extras[0]['id']])

    def test_human_link_extra_output_preserves_text_model_and_source(self):
        source, pred = rating(), prediction(source_locator='/fixture/run-output.json')
        reference, extras = self.resolve(source, [], [link(source, [pred])], [pred],
                                         [{'id': 'run-a', 'model': 'Declared original model'}])
        self.assertEqual(reference['sides'][0]['status'], 'verified')
        self.assertEqual(reference['sides'][0]['output_ids'], [extras[0]['id']])
        self.assertEqual(extras[0]['text'], pred['output_text'])
        self.assertEqual(extras[0]['model_label'], 'Declared original model')
        self.assertEqual(extras[0]['source_path'], '/fixture/run-output.json')
        self.assertIn('output_view', extras[0])
        json.dumps(extras, allow_nan=False)

    def test_verified_pair_uses_ordered_runs_and_ordered_hashes(self):
        source = pair()
        a, b = prediction('z-run', 'A answer'), prediction('a-run', 'B answer')
        reference, extras = self.resolve(source, [], [link(source, [a, b])], [b, a])
        by_id = {value['id']: value for value in extras}
        self.assertEqual([side['status'] for side in reference['sides']], ['verified', 'verified'])
        self.assertEqual([by_id[side['output_ids'][0]]['text'] for side in reference['sides']], ['A answer', 'B answer'])
        reversed_hashes = link(source, [a, b], prediction_hashes=[b['prediction_sha256'], a['prediction_sha256']])
        reference, extras = self.resolve(source, [], [reversed_hashes], [a, b])
        self.assertTrue(reference['warnings'])
        self.assertEqual([side['status'] for side in reference['sides']], ['unavailable', 'unavailable'])
        self.assertEqual(extras, [])

    def test_stale_missing_failed_unconfirmed_and_cross_frame_links_never_verify(self):
        source, pred = rating(), prediction()
        original = link(source, [pred])
        variants = [(dict(original, source_rating_hashes={}), [pred]),
                    (dict(original, source_rating_hashes={source['id']: '0' * 64}), [pred]),
                    (dict(original, human_confirmed=False), [pred]),
                    (dict(original, frame_id=OTHER), [pred]),
                    (dict(original, prediction_hashes=['0' * 64]), [pred]),
                    (original, []), (original, [dict(pred, status='failed')]),
                    (original, [dict(pred, output_text='Changed without updating hash')]),
                    (original, [dict(pred, frame_id=OTHER)]),
                    (dict(original, other_run_id='unexpected-run'), [pred])]
        for saved, predictions in variants:
            with self.subTest(saved=saved, predictions=predictions):
                reference, extras = self.resolve(source, [output()], [saved], predictions)
                self.assertEqual(reference['sides'][0]['status'], 'candidate')
                self.assertTrue(reference['warnings'])
                self.assertEqual(extras, [])

    def test_conflicting_current_human_links_do_not_select_first_or_latest(self):
        source = pair()
        a, b = prediction('run-a', 'A answer'), prediction('run-b', 'B answer')
        links = [link(source, [a, b]), link(source, [b, a], id='reversed-link')]
        reference, extras = self.resolve(source, [], links, [a, b])
        self.assertEqual([side['status'] for side in reference['sides']], ['conflict', 'conflict'])
        self.assertTrue(any('ordered rated outputs' in value for value in reference['warnings']))
        self.assertEqual(len(extras), 2)
        self.assertTrue(all(len(side['output_ids']) == 2 for side in reference['sides']))

    def test_current_human_link_can_verify_while_stale_link_remains_warned(self):
        source, pred = rating(), prediction()
        links = [link(source, [pred]), link(source, [pred], id='old-link', source_rating_hashes={})]
        reference, _ = self.resolve(source, [], links, [pred])
        self.assertEqual(reference['sides'][0]['status'], 'verified')
        self.assertTrue(any('old-link' in message for message in reference['warnings']))

    def test_human_link_does_not_silently_resolve_source_label_conflict(self):
        source, pred = rating(path_model_label='model-b'), prediction()
        reference, extras = self.resolve(source, [output(), output('out-b', 'model-b')], [link(source, [pred])], [pred])
        self.assertEqual(reference['sides'][0]['status'], 'conflict')
        self.assertEqual(set(reference['sides'][0]['output_ids']), {'out-a', 'out-b', extras[0]['id']})

    def test_resolver_is_pure_and_provider_payloads_do_not_receive_display_links(self):
        from tests.test_evaluation import entry
        from mousebench.providers import build_job_preview
        source, pred, descriptions = rating(), prediction(), [output()]
        args = (FRAME, [source], descriptions, [link(source, [pred])], [pred], [])
        original = copy.deepcopy(args)
        result = resolve_rating_references(*args)
        self.assertEqual(args, original)
        before = entry()
        after = dict(before, **result)
        for task in ('ground_truth', 'gt_check', 'inference', 'judge'):
            self.assertEqual(build_job_preview(task, [before]), build_job_preview(task, [after]))


class RatingReferenceStoreTests(unittest.TestCase):
    def test_frame_detail_adds_references_without_mutating_source_rows_or_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            try:
                source = rating()
                store.import_catalog({'frames': [{'id': FRAME}], 'ratings': [source]})
                store.register_text({'id': 'out-a', 'frame_id': FRAME, 'model_label': 'model-a'}, 'Fixture answer.', 'archive-hash')
                workflow = Workflow(store)
                run = store.import_run({'name': 'Fixture run', 'model': 'model-a', 'records': [{'frame_id': FRAME, 'output': 'Other verified fixture.'}]})
                workflow.verify_evidence_link({'frame_id': FRAME, 'run_id': run['id'], 'rating_ids': [source['id']],
                                               'human_confirmed': True, 'reviewer': 'Fixture expert', 'reason': 'Exact fixture correspondence.'})
                before = list(store.db.iterdump())
                detail = store.frame_detail(FRAME)
                self.assertEqual(detail['ratings'], [source])
                self.assertEqual(detail['descriptions'][0]['text'], 'Fixture answer.')
                refs = detail['rating_output_references'][source['id']]
                self.assertEqual(refs['sides'][0]['status'], 'verified')
                self.assertEqual(detail['rating_reference_outputs'][0]['text'], 'Other verified fixture.')
                self.assertEqual(list(store.db.iterdump()), before)
            finally:
                store.close()


if __name__ == '__main__':
    unittest.main()
