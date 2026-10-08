"""Collection imports preserve scientific records and exact source asset bytes."""
import hashlib
import http.client
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from pathlib import Path

import app
from mousebench.overview import build_overview
from mousebench.store import FIELDS, Store, dumps
from mousebench.workflow import Workflow


class CollectionTagsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_reimport_keeps_collection_metadata_and_approved_versions(self):
        self.store.import_catalog({'frames': [{'id': 'original', 'tags': [' v1.0 ', 'v1.0'],
                                               'collection_source': {'path': '/original'}}]})
        self.store.register_image('original', b'fixture source', '/original.jpg')
        self.store.save_input('original', {'prompt': 'Fixture prompt', 'reviewer': 'Fixture'})
        content = {'sections': {field: 'Fixture' for field in FIELDS}, 'facts': [
            {'id': 'f1', 'text': 'Fixture fact', 'field': 'Head', 'evidence': 'image', 'required': True}]}
        self.store.save_gt('original', {'status': 'reviewed', 'reviewer': 'Fixture', 'content': content})
        self.store.save_gt('original', {'status': 'approved', 'reviewer': 'Fixture', 'content': content,
                                       'expected_revision': 1, 'human_confirmed': True})
        versions = list(self.store.db.execute('SELECT kind,frame_id,revision,data FROM versions'))
        self.store.import_catalog({'frames': [{'id': 'original', 'recording_id': 'updated'},
                                               {'id': 'addition', 'tags': ['v1.1']}]})
        self.store.import_catalog({'frames': [{'id': 'original', 'tags': ['review-pilot']}]})
        frame = self.store.frame('original')
        self.assertEqual(frame['tags'], ['review-pilot', 'v1.0'])
        self.assertEqual(frame['collection_source'], {'path': '/original'})
        self.assertEqual(frame['recording_id'], 'updated')
        self.assertEqual(versions, list(self.store.db.execute('SELECT kind,frame_id,revision,data FROM versions')))
        self.assertEqual(self.store.frame_list(tag='v1.0')[0]['gt_status'], 'approved')
        self.assertEqual([f['id'] for f in self.store.frame_list(tag='v1.1')], ['addition'])
        self.assertEqual(self.store.frame_list(q='original', tag='v1.1'), [])
        self.assertEqual(self.store.frame_list(status='approved', tag='v1.1'), [])

    def test_summary_overview_and_catalog_export_retain_distinct_membership(self):
        self.store.import_catalog({'frames': [{'id': 'a', 'tags': ['v1.0']},
                                               {'id': 'b', 'tags': ['v1.1']}, {'id': 'future'}]})
        expected = [{'tag': 'v1.0', 'frames': 1}, {'tag': 'v1.1', 'frames': 1}]
        self.assertEqual(self.store.summary()['collection_tags'], expected)
        self.assertEqual(build_overview(self.store, Workflow(self.store))['dataset']['collection_tags'], expected)
        exported = {r['id']: r for r in self.store.export('catalog') if r['record_kind'] == 'frames'}
        self.assertEqual(exported['b']['tags'], ['v1.1'])
        self.assertEqual(exported['future']['tags'], [])
        self.assertEqual(self.store.frame_list(tag='unrecognized'), [])

    def test_invalid_tag_import_is_atomic(self):
        for bad in ('v1.0', [''], [1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.store.import_catalog({'frames': [{'id': 'ok', 'tags': ['v1.0']}, {'id': 'bad', 'tags': bad}]})
            self.assertEqual(self.store.get_records('frames'), [])

    def test_single_view_image_layout_survives_manual_input_without_changing_default(self):
        self.store.import_catalog({'frames': [{'id': 'single'}, {'id': 'legacy'}]})
        source = self.store.register_image('single', b'single fixture', '/single.jpg',
                                           kind='source_preview', image_layout='single_view_unmodified')
        self.store.register_image('legacy', b'concat fixture', '/concat.jpg')
        single = self.store.save_input('single', {'prompt': 'Fixture', 'reviewer': 'Fixture'})
        legacy = self.store.save_input('legacy', {'prompt': 'Fixture', 'reviewer': 'Fixture'})
        self.assertEqual(single['image_layout'], 'single_view_unmodified')
        self.assertEqual(single['image_sha256'], source['sha256'])
        self.assertEqual(legacy['image_layout'], 'original_concat_unmodified')

    def test_expanded_catalog_can_select_and_claim_more_than_395_frames(self):
        frames = [{'id': f'fixture-{index:03}'} for index in range(435)]
        self.store.import_catalog({'frames': frames})
        workflow = Workflow(self.store)
        self.assertEqual(len(workflow.frame_ids(None)), 435)
        with self.assertRaises(ValueError): workflow.frame_ids([frames[0]['id']] * 2)
        with self.store.transaction():
            workflow.put('batch', {'id': 'fixture-batch', 'task': 'inference'})
            for frame in frames:
                workflow.save_job({'id': 'job-'+frame['id'], 'batch_id': 'fixture-batch',
                    'frame_id': frame['id'], 'status': 'pending', 'attempts': 0, 'history': []})
        # This fixture isolates catalog-sized lease limits. Other workflow
        # tests exercise the current-input/GT checks with complete packets.
        with patch.object(workflow, '_current') as current:
            self.assertEqual(len(workflow.claim('fixture-batch', 'Fixture worker', 435)['jobs']), 435)
            self.assertEqual(current.call_count, 435)
        with self.assertRaises(ValueError): workflow.claim('fixture-batch', 'Fixture worker', 436)


class QuietHandler(app.Handler):
    def log_message(self, *args):
        pass


class CollectionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = Store(self.root)
        self.store.import_catalog({'frames': [{'id': 'first', 'tags': ['v1.0']}, {'id': 'second', 'tags': ['v1.1']}]})
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        self.server.data_dir = self.root
        self.server.csrf_token = 'fixture-token'
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.store.close()
        self.temp.cleanup()

    def get(self, path):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            conn.request('GET', path)
            response = conn.getresponse()
            return response.status, response.read(), response.getheader('Content-Type')
        finally:
            conn.close()

    def asset(self, asset_id, path, raw, kind='assets'):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        record = {'id': asset_id, 'frame_id': 'second', 'path': path,
                  'sha256': hashlib.sha256(raw).hexdigest()}
        with self.store.transaction():
            self.store.db.execute('INSERT INTO records VALUES(?,?,?,?)', (kind, asset_id, 'second', dumps(record)))
        return target

    def test_frame_endpoint_filters_collection(self):
        status, raw, _ = self.get('/api/frames?tag=v1.1')
        self.assertEqual(status, 200)
        self.assertEqual([f['id'] for f in json.loads(raw)], ['second'])
        self.assertEqual(json.loads(self.get('/api/frames?tag=v1.0&q=second')[1]), [])

    def test_rendered_output_and_raw_log_preserve_bytes_and_check_integrity(self):
        raw = b'\xff\xd8fixture rendered output'
        target = self.asset('rendered', 'objects/rendered.jpg', raw)
        self.assertEqual(self.get('/api/assets/rendered'), (200, raw, 'image/jpeg'))
        target.write_bytes(b'changed source')
        self.assertEqual(self.get('/api/assets/rendered')[0], 400)
        log = b'{"source":"fixture"}\n'
        self.asset('log', 'objects/log.json', log, 'raw_records')
        self.assertEqual(self.get('/api/assets/log'), (200, log, 'application/json; charset=utf-8'))

    def test_unknown_types_and_outside_paths_are_not_served(self):
        self.asset('unsafe', 'objects/unsafe.html', b'<script>fixture</script>')
        self.assertEqual(self.get('/api/assets/unsafe')[0], 400)
        with self.store.transaction():
            self.store.db.execute('INSERT INTO records VALUES(?,?,?,?)', ('assets', 'outside', 'second', dumps({
                'path': '/etc/passwd', 'sha256': 'untrusted'})))
        self.assertEqual(self.get('/api/assets/outside')[0], 404)


if __name__ == '__main__':
    unittest.main()
