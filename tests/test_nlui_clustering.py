"""Offline coverage, export and provider-failure regressions for NLUI clustering."""
import csv
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '4_NLUI'))
from tools_function import cluster_utils as clustering
from tools_function import provider
from tools_function.cluster import cluster


def response(merge=False, cluster_A='resting', cluster_B='walking'):
    return {'choices': [{'message': {'content': json.dumps({
        'similarity_score': 0.5, 'merge': merge, 'cluster_A': cluster_A,
        'cluster_B': cluster_B, 'reason': 'Example {with braces}',
    })}}]}


def merged(ids, name='walking', value=True):
    return {'level': 1, 'cluster_curr': name, 'message': {'merge': value, 'ids': ids}}


@pytest.mark.parametrize('value, expected', [(True, True), (False, False), ('True', True), ('false', False), (' FALSE ', False)])
def test_parse_boolean_and_legacy_string(value, expected):
    assert clustering.parse_decision(response(value))['merge'] is expected


@pytest.mark.parametrize('value', [1, 0, None, 'yes', [], {}])
def test_invalid_merge_is_rejected(value):
    with pytest.raises(ValueError):
        clustering.parse_decision(response(value))


def test_fenced_json_supports_braces_inside_reason():
    result = response(True)
    content = result['choices'][0]['message']['content']
    result['choices'][0]['message']['content'] = f'Decision:\n```json\n{content}\n```'
    assert clustering.parse_decision(result)['reason'] == 'Example {with braces}'


def test_traversal_retains_leaves_and_positional_descriptions(monkeypatch):
    replies = []
    def split(system_prompt, a, b, api):
        replies.append((a, b))
        return response(False)
    monkeypatch.setattr(clustering, 'gpt_request', split)
    tree = to_tree(linkage([[0], [1], [10], [11]]))
    # Nonconsecutive integer indexes must not be mistaken for row positions.
    descriptions = pd.Series(['a', 'b', 'c', 'd'], index=[8, 16, 24, 32])
    decisions = clustering.bfs_traverse_print(tree, 'prompt', 'key', descriptions, ['a', 'b', 'c', 'd'])
    terminals = [entry for entry in decisions.values() if entry.get('terminal_reason') == 'leaf']
    assert sorted(index for entry in terminals for index in entry['message']['ids']) == [0, 1, 2, 3]
    labels, behaviors = clustering.save_des_labels(decisions, 4)
    assert labels.tolist() == [-1, -1, -1, -1]
    assert behaviors == []
    assert len(replies) == 3


def test_depth_limit_records_unassigned_and_makes_no_extra_calls(monkeypatch):
    request = Mock(return_value=response('false'))
    monkeypatch.setattr(clustering, 'gpt_request', request)
    tree = to_tree(linkage([[0], [1], [10], [11]]))
    decisions = clustering.bfs_traverse_print(tree, 'prompt', 'key', ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], max_level=0)
    assert request.call_count == 1
    terminals = [entry for entry in decisions.values() if entry.get('terminal_reason')]
    assert sorted(index for entry in terminals for index in entry['message']['ids']) == [0, 1, 2, 3]
    assert {entry['terminal_reason'] for entry in terminals} == {'depth_limit'}
    assert clustering.save_des_labels(decisions, 4)[0].tolist() == [-1] * 4


def test_root_merge_has_neutral_label_and_all_frames(monkeypatch):
    monkeypatch.setattr(clustering, 'gpt_request', Mock(return_value=response(True)))
    tree = to_tree(linkage([[0], [1], [2]]))
    decisions = clustering.bfs_traverse_print(tree, 'prompt', 'key', ['a', 'b', 'c'], ['a', 'b', 'c'])
    labels, behaviors = clustering.save_des_labels(decisions, 3)
    assert labels.tolist() == [0, 0, 0]
    assert behaviors == [f'Cluster {tree.id}']


def test_single_leaf_never_calls_provider(monkeypatch):
    request = Mock(side_effect=AssertionError('Network must not be called for one leaf'))
    monkeypatch.setattr(clustering, 'gpt_request', request)
    decisions = clustering.bfs_traverse_print(ClusterNode(0), 'prompt', 'key', ['a'], ['a'])
    assert clustering.save_des_labels(decisions, 1)[0].tolist() == [-1]


def test_unassigned_does_not_inherit_first_behavior():
    labels, behaviors = clustering.save_des_labels({2: merged([1])}, 3)
    assert labels.tolist() == [-1, 0, -1]
    assert behaviors == ['walking']


@pytest.mark.parametrize('ids', [[-1], [3], [True], [0.5], ['1'], [1, 1], []])
def test_invalid_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        clustering.save_des_labels({0: merged(ids)}, 3)


def test_overlapping_terminal_positions_are_rejected():
    with pytest.raises(ValueError, match='overlapping'):
        clustering.save_des_labels({0: merged([1]), 1: merged([1, 2])}, 3)


def test_more_than_255_clusters_do_not_wrap():
    labels, behaviors = clustering.save_des_labels({index: merged([index], str(index)) for index in range(300)}, 300)
    assert labels.dtype == np.dtype('int64')
    assert labels.tolist() == list(range(300))
    assert len(behaviors) == 300


def test_export_preserves_frame_ids_unassigned_and_quoted_names(tmp_path):
    path = tmp_path / 'nested' / 'cluster.csv'
    proportions = clustering.export_cluster_labels([8, 24, 72, 96], np.array([-1, 0, 1, 2]), ['walking, slow', 'resting', 'resting'], path)
    with path.open(newline='') as file:
        assert list(csv.reader(file)) == [
            ['frame_id', 'behavior'], ['00000008', 'Unassigned'],
            ['00000024', 'walking, slow'], ['00000072', 'resting'], ['00000096', 'resting'],
        ]
    assert proportions == {'Unassigned': 0.25, 'walking, slow': 0.25, 'resting': 0.5}
    assert sum(proportions.values()) == 1


@pytest.mark.parametrize('labels', [[-2], [1], [True], [0.5]])
def test_export_rejects_invalid_labels(labels, tmp_path):
    with pytest.raises(ValueError):
        clustering.export_cluster_labels([8], labels, ['walking'], tmp_path / 'cluster.csv')


def test_malformed_provider_response_is_bounded_and_timed_out(monkeypatch):
    fake = Mock(status_code=200)
    fake.json.return_value = {'choices': []}
    post = Mock(return_value=fake)
    monkeypatch.setattr(provider.requests, 'post', post)
    monkeypatch.setattr(provider.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        clustering.gpt_request('prompt', ['a'], ['b'], 'not-a-real-key')
    assert post.call_count == 3
    assert post.call_args.kwargs['timeout'] == (10, 60)


def test_authentication_failure_is_not_retried(monkeypatch):
    post = Mock(return_value=Mock(status_code=401))
    monkeypatch.setattr(provider.requests, 'post', post)
    with pytest.raises(RuntimeError, match='HTTP 401'):
        clustering.gpt_request('prompt', ['a'], ['b'], 'not-a-real-key')
    assert post.call_count == 1


def test_transient_provider_error_then_boolean_success(monkeypatch):
    good = Mock(status_code=200)
    good.json.return_value = response(False)
    post = Mock(side_effect=[Mock(status_code=429), good])
    monkeypatch.setattr(provider.requests, 'post', post)
    monkeypatch.setattr(provider.time, 'sleep', lambda _: None)
    assert clustering.parse_decision(clustering.gpt_request('prompt', ['a'], ['b'], 'key'))['merge'] is False
    assert post.call_count == 2


def test_embedding_exhaustion_raises_instead_of_unbound_response(monkeypatch):
    post = Mock(side_effect=provider.requests.Timeout('offline test'))
    monkeypatch.setattr(provider.requests, 'post', post)
    monkeypatch.setattr(provider.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        provider.encode_string_to_feature('walking', 'key')
    assert post.call_count == 3


def test_cluster_rejects_misaligned_ids_before_api_request(tmp_path, monkeypatch):
    path = tmp_path / 'misaligned.h5'
    with pd.HDFStore(path, 'w') as store:
        store.put('features_des', pd.DataFrame([[0.0], [1.0]], index=[8, 16]))
        store.put('descriptions', pd.DataFrame({'id': [16, 8], 'description': ['a', 'b']}))
        store.put('linkage_matrix', pd.DataFrame(linkage([[0.0], [1.0]])))
    monkeypatch.setattr(clustering, 'gpt_request', Mock(side_effect=AssertionError('No network expected')))
    with pytest.raises(ValueError, match='same row order'):
        cluster(path, tmp_path, 'key')


def test_search_rejects_misaligned_ids_before_query(tmp_path, monkeypatch):
    from tools_function import search as searching
    path = tmp_path / 'misaligned_search.h5'
    with pd.HDFStore(path, 'w') as store:
        store.put('features_des', pd.DataFrame([[1., 0.], [0., 1.]], index=[8, 16]))
        store.put('descriptions', pd.DataFrame({'id': [16, 8], 'description': ['b', 'a']}))
    provider_call = Mock(side_effect=AssertionError('No network expected'))
    monkeypatch.setattr(searching, 'encode_string_to_feature', provider_call)
    with pytest.raises(ValueError, match='same row order'):
        searching.search(path, 'walking', 'key')
    provider_call.assert_not_called()


@pytest.mark.parametrize('bad', [[0., 0.], [float('nan'), 1.], [float('inf'), 1.]])
@pytest.mark.parametrize('operation', ['search', 'classification'])
def test_similarity_tools_reject_invalid_stored_vectors_before_query(tmp_path, monkeypatch, bad, operation):
    from tools_function import search as searching, classification as classifying
    module = searching if operation == 'search' else classifying
    path = tmp_path / 'bad_vectors.h5'
    with pd.HDFStore(path, 'w') as store:
        store.put('features_des', pd.DataFrame([[1., 0.], bad], index=[8, 16]))
        store.put('descriptions', pd.DataFrame({'id': [8, 16], 'description': ['a', 'b']}))
    provider_call = Mock(side_effect=AssertionError('No network expected'))
    monkeypatch.setattr(module, 'encode_string_to_feature', provider_call)
    with pytest.raises(ValueError, match='nonfinite|nonzero'):
        getattr(module, operation)(path, 'walking' if operation == 'search' else ['walking'], 'key')
    provider_call.assert_not_called()


def test_search_and_classification_return_matching_stored_ids(tmp_path, monkeypatch):
    from tools_function import search as searching, classification as classifying
    path = tmp_path / 'valid_vectors.h5'
    with pd.HDFStore(path, 'w') as store:
        store.put('features_des', pd.DataFrame([[1., 0.], [0., 1.]], index=[8, 16]))
        store.put('descriptions', pd.DataFrame({'id': [8, 16], 'description': ['a', 'b']}))
    monkeypatch.setattr(searching, 'encode_string_to_feature', Mock(return_value=[0., 1.]))
    ids, similarities = searching.search(path, 'walking', 'key')
    assert ids == ['00000016', '00000008']
    np.testing.assert_allclose(similarities, [1., 0.])
    monkeypatch.setattr(classifying, 'encode_string_to_feature', Mock(side_effect=[[1., 0.], [0., 1.]]))
    assert classifying.classification(path, ['resting', 'walking'], 'key') == {'00000008': 0, '00000016': 1}


def test_similarity_tools_reject_query_dimension_mismatch():
    from tools_function import search as searching, classification as classifying
    with pytest.raises(ValueError, match='dimensions'):
        searching.calculate_cosine_similarity([1., 0., 0.], [[1., 0.]])
    with pytest.raises(ValueError, match='dimensions'):
        classifying.create_similarity_matrix([[1., 0.]], [[1., 0., 0.]])


def test_classification_tool_exports_quoted_labels_and_aggregates_duplicates(tmp_path):
    import ast
    from datetime import datetime
    from glob import glob
    import os
    source = Path(__file__).resolve().parents[1] / '4_NLUI/tools.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'tool_classification')
    function.decorator_list = []
    namespace = {
        'os': os, 'datetime': datetime, 'glob': glob,
        'get_api_key': lambda: 'offline-key',
        'tool_config': {'file_path': 'not-opened.h5', 'img_folder': str(tmp_path), 'workspace': str(tmp_path / 'output')},
        'classification_func': Mock(return_value={'00000008': 0, '00000016': 1, '00000024': 2}),
        'export_cluster_labels': clustering.export_cluster_labels,
    }
    # Run the real tool body with its provider boundary mocked; LangChain is unnecessary.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    result = namespace['tool_classification'](['walking, slowly', 'resting "still"', 'resting "still"'])
    with Path(result['save_path']).open(newline='') as handle:
        assert list(csv.reader(handle)) == [
            ['frame_id', 'behavior'], ['00000008', 'walking, slowly'],
            ['00000016', 'resting "still"'], ['00000024', 'resting "still"'],
        ]
    assert result['static'] == {'walking, slowly': 1 / 3, 'resting "still"': 2 / 3}
