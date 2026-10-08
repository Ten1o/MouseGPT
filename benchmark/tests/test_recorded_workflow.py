"""Logged input import, batching and release integration; fixture data only."""
import copy
import hashlib
import json
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import workflow as cli
from mousebench.store import FIELDS
from mousebench.workflow import Workflow
from mousebench.protocol import default_rubric
from mousebench.api_providers import _payload, validate_config
import test_recorded_inputs as fixtures
FRAME, OTHER = fixtures.FRAME, fixtures.OTHER


class RecordedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RecordedInputTests()
        self.fixture.setUp()
        self.store = self.fixture.store
        self.wf = Workflow(self.store)

    def tearDown(self):
        self.fixture.tearDown()

    def freeze(self, frames=None):
        return self.wf.dispatch({'action':'freeze_recorded_inputs','frame_ids':frames or [FRAME],
                                'reviewer':'Fixture archive importer'})

    def test_per_frame_inputs_are_distinct_and_existing_versions_stay(self):
        self.store.register_image(OTHER, b'other concat', '/fixture/other.jpg')
        source = copy.deepcopy(self.fixture.source)
        source['messages'][1]['content'][0]['text'] = source['messages'][1]['content'][0]['text'].replace('9.77','1.25')
        self.fixture.add_source('log-other', source, frame=OTHER)
        self.assertEqual(self.freeze([FRAME,OTHER])['saved'], 2)
        self.assertIn('9.77', self.store.latest('input',FRAME)['prompt'])
        self.assertIn('1.25', self.store.latest('input',OTHER)['prompt'])
        self.assertNotEqual(self.store.latest('input',FRAME)['input_hash'], self.store.latest('input',OTHER)['input_hash'])
        self.assertEqual(self.freeze([FRAME,OTHER])['skipped'], 2)
        self.assertEqual(len(self.store.versions('input',FRAME)), 1)
        self.assertEqual(self.store.versions('gt',FRAME), [])
        self.assertEqual(self.wf.records('batch'), [])

    def test_missing_log_rolls_back_entire_input_import(self):
        with self.assertRaises(ValueError): self.freeze([FRAME,OTHER])
        self.assertIsNone(self.store.latest('input',FRAME))
        self.assertEqual(self.wf.records('event'), [])

    def test_imported_source_round_trips_through_actual_job_transport(self):
        self.freeze()
        batch = self.wf.create_batch({'task':'inference','executor':'codex','model':'fixture','frame_ids':[FRAME]})
        job = self.wf.job(batch['jobs'][0]['id'])['packet']
        config = validate_config({'provider':'openai','model':'fixture','api_mode':'chat_completions'})
        request, _, _ = _payload(job, config, self.store.root)
        self.assertEqual(request['messages'], self.fixture.source['messages'])
        self.assertEqual(len(job['attachment_manifest']), 2)
        self.assertNotIn('secret_output', json.dumps(request))

    def test_release_archive_contains_original_input_views(self):
        self.freeze()
        content = {'sections':{key:'Fixture fact only.' for key in FIELDS},'facts':[
            {'id':'f1','text':'Fixture fact.','field':'Overall','required':True,'evidence':'image'}], 'uncertainties':[]}
        self.store.save_gt(FRAME,{'reviewer':'Fixture human','content':content,'status':'reviewed','expected_revision':0})
        self.store.save_gt(FRAME,{'reviewer':'Fixture human','content':content,'status':'approved','human_confirmed':True,'expected_revision':1})
        rubric=self.wf.save_rubric({'content':default_rubric(),'author':'Fixture'})
        self.wf.freeze_rubric({'id':rubric['id'],'reviewer':'Fixture'})
        release=self.wf.create_release({'frame_ids':[FRAME],'rubric_id':rubric['id'],'reviewer':'Fixture','name':'Fixture only'})
        output=self.store.root/'release.zip'
        with patch('builtins.print'):
            cli.main(['--data',str(self.store.root),'export-release',release['id'],'--output',str(output)])
        with zipfile.ZipFile(output) as z:
            for image in self.store.latest('input',FRAME)['input_images']:
                self.assertEqual(hashlib.sha256(z.read(image['path'])).hexdigest(),image['sha256'])
